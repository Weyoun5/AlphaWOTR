// The module's API for the worker (docs/AlphaWOTR-Wasm-Plan.md section 2): one bot per handle, synchronous.
//
//   aw_create(config_json) -> handle (null: aw_last_error())
//       {"dir": the folder holding adapter.json, prototypes.awg, tables.awg and the nets (net_fp.awn, ...),
//        "sims": the simulations of a full decision (300), "max_seconds": a cap per decision (none)}
//   aw_message(handle, line) -> the reply line ('' none; valid until the next call)
//   aw_destroy(handle)
// A new_game starts a new object model for the game (the adapter's states of the last game are freed).
#include "bot.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>

namespace {
std::string last_error;
std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
struct Instance {
    awad::Tables tables;
    pyobj::Graph protos;
    awdec::Box box;
    std::unique_ptr<pyobj::Model> model;
    std::unique_ptr<awad::Adapter> adapter;
    std::unique_ptr<awbot::Bot> bot;
    int64_t sims = 300;
    std::optional<double> max_seconds;
    std::string out;
    std::map<std::string, double> rates;
    void fresh() {   // a new object model and bot (the measured rates and the options carried over)
        if (bot) { rates = bot->rates; sims = bot->sims; }
        bot.reset(); adapter.reset();
        model = std::make_unique<pyobj::Model>();
        pyobj::Obj* p = model->decode(protos);
        adapter = std::make_unique<awad::Adapter>(*model, tables, p);
        adapter->obs_tables = box.obs;
        bot = std::make_unique<awbot::Bot>(*model, *adapter, tables, box);
        bot->sims = sims; bot->max_seconds = max_seconds; bot->rates = rates;
    }
};
}  // namespace

extern "C" {
const char* aw_last_error() { return last_error.c_str(); }

void* aw_create(const char* config_json) {
    try {
        const awjson::Value cfg = awjson::parse(config_json ? config_json : "{}");
        const std::string dir = cfg.get("dir") ? cfg.get("dir")->s : ".";
        auto in = std::make_unique<Instance>();
        auto tj = slurp(dir + "/adapter.json");
        in->tables.load(awjson::parse(std::string(tj.begin(), tj.end())));
        auto pg = slurp(dir + "/prototypes.awg");
        uint32_t nl; std::memcpy(&nl, &pg[8], 4);
        in->protos = pyobj::read_graph(&pg[12 + nl]);
        std::string err;
        if (!in->box.load(dir + "/tables.awg", dir, err)) throw std::runtime_error(err);
        if (cfg.get("sims")) in->sims = cfg.get("sims")->as_int();
        if (cfg.get("max_seconds") && !cfg.get("max_seconds")->is_null())
            in->max_seconds = cfg.get("max_seconds")->kind == awjson::Value::Real ? cfg.get("max_seconds")->d : double(cfg.get("max_seconds")->as_int());
        in->fresh();
        return in.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

const char* aw_message(void* handle, const char* line) {
    auto* in = static_cast<Instance*>(handle);
    try {
        const awjson::Value m = awjson::parse(line);
        if (m.get("type") && m.get("type")->s == "new_game") in->fresh();
        in->out = in->bot->handle(m);
    } catch (const std::exception& e) { last_error = e.what(); in->out.clear(); }
    return in->out.c_str();
}

void aw_destroy(void* handle) { delete static_cast<Instance*>(handle); }
}
