// Phase 4: the bot on the module over whole transcripts, against the Python adapter's answers.
//
//   bot_parity FIXTURES_DIR TABLES_DIR [--verbose] [--max N]
//
// FIXTURES_DIR: adapter.json, prototypes.awg and transcripts.jsonl (record_build_state.py --log: every message the
// Python bot handled, each decide with its answer, how, the searches it ran and whether a plan or a chain followed);
// TABLES_DIR: tables.awg and the nets. Each transcript is played through one Bot, message by message, with the
// Python's search arguments. A decision the port cannot answer yet (a chain of the engine's work copy) takes the
// Python's answer and leaves the bot out of step until the Python bot has neither a plan nor a chain again: the
// decisions in step are compared.
#include "bot.h"
#include "graph_text.h"
#include "wotr_random.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>

namespace {
std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: bot_parity FIXTURES_DIR TABLES_DIR [--verbose] [--max N]\n"); return 2; }
    const std::string dir = argv[1], tdir = argv[2];
    bool verbose = false; long max_dec = -1;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose") verbose = true;
        else if (std::string(argv[i]) == "--max" && i + 1 < argc) max_dec = std::atol(argv[++i]);
    }
    auto tj = slurp(dir + "/adapter.json");
    awad::Tables tables;
    tables.load(awjson::parse(std::string(tj.begin(), tj.end())));
    auto pg = slurp(dir + "/prototypes.awg");
    uint32_t nl; std::memcpy(&nl, &pg[8], 4);
    const pyobj::Graph pgraph = pyobj::read_graph(&pg[12 + nl]);
    awdec::Box box;
    std::string err;
    const std::string nets = std::ifstream(tdir + "/net_fp.awn").good() ? tdir : tdir + "/..";
    if (!box.load(tdir + "/tables.awg", nets, err)) { std::fprintf(stderr, "box: %s\n", err.c_str()); return 2; }

    std::ifstream in(dir + "/transcripts.jsonl");
    std::string line, current;
    std::unique_ptr<pyobj::Model> model;
    std::unique_ptr<awad::Adapter> adapter;
    std::unique_ptr<awbot::Bot> bot;
    bool in_step = true, resync = false;
    size_t applies_off_n = 0, total_drew = 0;
    size_t total = 0, stepped = 0, same = 0, unported = 0, games = 0, search_args_off = 0;
    std::map<std::string, std::pair<size_t, size_t>> by_how;      // Python's how -> (in step, same)
    std::map<std::string, size_t> unported_why, cpp_how_diff;
    size_t shown = 0;
    double secs = 0;
    while (std::getline(in, line)) {
        const awjson::Value item = awjson::parse(line);
        const std::string tr = item.get("transcript") ? item.get("transcript")->s : "";
        if (tr != current) {   // a new transcript: a new bot (and object model)
            current = tr; ++games;
            bot.reset(); adapter.reset();
            model = std::make_unique<pyobj::Model>();
            pyobj::Obj* protos = model->decode(pgraph);
            adapter = std::make_unique<awad::Adapter>(*model, tables, protos);
            adapter->obs_tables = box.obs;
            bot = std::make_unique<awbot::Bot>(*model, *adapter, tables, box);
            in_step = true;
        }
        const awjson::Value& m = *item.get("m");
        const bool decide = m.get("type") && m.get("type")->s == "decide";
        if (!decide) { bot->handle(m); continue; }
        if (max_dec >= 0 && long(total) >= max_dec) break;
        ++total;
        // the Python's searches, in order
        std::deque<awbot::SearchArgs> args;
        if (const auto* ss = item.get("searches"))
            for (const auto& s : ss->arr) args.push_back({uint64_t(s.arr[0].as_int()), uint32_t(s.arr[1].as_int()), uint64_t(s.arr[2].as_int()), s.arr[3].as_int()});
        // out of step: the memo's Random as the Python's before this decision (a chain the port skipped may have
        // left it elsewhere: the Python rebuilt the state there)
        if (resync && item.get("rng") && !item.get("rng")->is_null()) {
            const auto& r = *item.get("rng");
            std::vector<uint32_t> words;
            for (size_t i = 0; i + 1 < r.arr[0].arr.size(); ++i) words.push_back(uint32_t(r.arr[0].arr[i].as_int()));
            const uint32_t index = uint32_t(r.arr[0].arr.back().as_int());
            const bool hg = !r.arr[1].is_null();
            bot->set_memo_rng(words, index, hg, hg ? (r.arr[1].kind == awjson::Value::Real ? r.arr[1].d : double(r.arr[1].as_int())) : 0.0);
            resync = false;
        }
        // Python's global Random before each engine action on the work copy, in order
        std::deque<const awjson::Value*> applies;
        if (const auto* ap = item.get("applies")) for (const auto& x : ap->arr) applies.push_back(&x);
        size_t applies_drew = 0;
        bool applies_off = false;
        bot->next_apply_rng = [&]() -> void* {
            if (applies.empty()) { applies_off = true; return nullptr; }
            const awjson::Value& x = *applies.front(); applies.pop_front();
            applies_drew += x.get("drew") && x.get("drew")->truthy();
            const auto& r = *x.get("rng");
            std::vector<uint32_t> words;
            for (size_t i = 0; i + 1 < r.arr[0].arr.size(); ++i) words.push_back(uint32_t(r.arr[0].arr[i].as_int()));
            const bool hg = !r.arr[1].is_null();
            const double g = hg ? (r.arr[1].kind == awjson::Value::Real ? r.arr[1].d : double(r.arr[1].as_int())) : 0.0;
            uint64_t bits = 0; std::memcpy(&bits, &g, 8);
            return wotr_random_create(words.data(), words.size(), uint32_t(r.arr[0].arr.back().as_int()), hg ? 1 : 0, bits);
        };
        bool args_off = false;
        bot->next_search = [&](int64_t) { if (args.empty()) { args_off = true; return awbot::SearchArgs{0, 0, 1, 8}; } auto a = args.front(); args.pop_front(); return a; };
        const auto t0 = std::chrono::steady_clock::now();
        bot->handle(m);
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto& tr_ = bot->last;
        const int64_t want = item.get("o") && !item.get("o")->is_null() ? item.get("o")->as_int() : -1;
        const std::string how = item.get("how") && item.get("how")->kind == awjson::Value::Str ? item.get("how")->s : "?";
        const bool py_plan = item.get("plan") && !item.get("plan")->is_null();
        const bool py_work = item.get("work") && item.get("work")->truthy();
        if (!tr_.unported.empty()) {
            ++unported; ++unported_why[tr_.unported];
            in_step = false;
        } else if (in_step) {
            ++stepped;
            const bool ok = tr_.o && *tr_.o == want;
            same += ok; auto& bh = by_how[how]; ++bh.first; bh.second += ok;
            if (args_off || !args.empty()) ++search_args_off;
            if (applies_off || !applies.empty()) ++applies_off_n;
            total_drew += applies_drew;
            if (tr_.how != how) ++cpp_how_diff[how + " -> " + tr_.how];
            if (!ok && (verbose || shown < 20)) {
                ++shown;
                const auto* dec = m.get("decision");
                std::printf("%s #%zu %s: C++ %lld (%s%s%s), Python %lld (%s)%s\n", tr.c_str(), total, dec && dec->get("kind") ? dec->get("kind")->s.c_str() : "?",
                            tr_.o ? (long long)*tr_.o : -1LL, tr_.how.c_str(), tr_.note.empty() ? "" : ", ", tr_.note.c_str(), (long long)want, how.c_str(),
                            args_off ? " [more searches than Python's]" : !args.empty() ? " [fewer searches than Python's]" : "");
            }
            if (!ok) in_step = false;
        }
        if (!in_step && !py_plan && !py_work) {   // the Python bot has neither a plan nor a chain: in step again
            bot->drop_plan(); bot->drop_work(); in_step = true; resync = true;
        }
    }
    std::printf("%s: %zu games, %zu decisions; %zu in step, %zu of them the Python's answer; %zu not ported (",
                same == stepped ? "PASS" : "DIFF", games, total, stepped, same, unported);
    for (const auto& [k, v] : unported_why) std::printf("%s %zu; ", k.c_str(), v);
    std::printf("); by how:");
    for (const auto& [k, v] : by_how) std::printf(" %s %zu/%zu", k.c_str(), v.second, v.first);
    std::printf("; how differs:");
    for (const auto& [k, v] : cpp_how_diff) std::printf(" [%s] %zu", k.c_str(), v);
    std::printf("; engine actions on the copy off %zu (Python's drew from its Random %zu times)", applies_off_n, total_drew);
    std::printf("; search arguments off %zu; %.1f ms a decision\n", search_args_off, total ? secs / total * 1e3 : 0.0);
    return same == stepped ? 0 : 1;
}
