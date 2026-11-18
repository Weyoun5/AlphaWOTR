// The AlphaWOTR bot on the module: protocol/wotr_protocol_bot.py's AlphaWotrBot (handle, decide, _decide,
// _search_and_map, _apply_copy, _chain, chain_fits, chain_over, diff_answer) and LeagueSearcher.choose at strength 1,
// over the adapter's port (adapter.h), the plans (plans.h) and the decision on the box (driver/decision.h).
#pragma once
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "adapter.h"
#include "decision.h"
#include "plans.h"

namespace awbot {

// one search: the seat's seed, its search index, the search stream's seed, the simulations
struct SearchArgs { uint64_t seat = 0; uint32_t index = 0; uint64_t seed = 0; int64_t sims = 0; };

// AlphaWotrBot._decision_ms(clock): the time per decision the search may use (none: no limit but the simulations)
std::optional<double> decision_ms(const awjson::Value& clock, const std::string& clock_kind);

struct Menu;

class Bot {
public:
    Bot(pyobj::Model& model, awad::Adapter& adapter, const awad::Tables& tables, awdec::Box& box);
    ~Bot();
    // a referee message -> the reply as JSON ("" none)
    std::string handle(const awjson::Value& m);

    // LeagueSearcher: the simulations of a full decision (--sims), max_seconds, the measured rates per kind
    int64_t sims = 300;
    std::optional<double> max_seconds;
    std::map<std::string, double> rates;
    std::map<std::string, std::vector<double>> recent;
    // the next search's arguments, given its simulations (the harness: the recorded ones; else the bot's own)
    std::function<SearchArgs(int64_t sims)> next_search;
    // the Random the next engine action on the work copy draws from (a wotr_random handle, owned by the caller; the
    // harness: Python's global Random before it; null or unset: the bot's own stream)
    std::function<void*()> next_apply_rng;
    // LeagueSearcher.budget(kind, left_ms)
    int64_t budget(const std::string& kind, std::optional<double> left_ms) const;
    // The certain line's time (seconds; <= 0 none): this port's own cap in the browser. Their Python bounds it by
    // counts and 15 s (certain_line.SECONDS), and on a per-decision clock by the answer's deadline less 40 ms
    // (certain_line.DEADLINE, the tournament adapter); the port keeps both and adds this cap, so a line the engine
    // searches long cannot hold a decision up for seconds (profiling the tournament bot, 2026-11-18: up to 0.8 s).
    double certain_seconds = 1.0;
    // false: as before the 2026-11-18 speed fixes (the search works the menu out again; the certain line bounded by
    // its counts and 15 s only): the measurements' "before"
    bool speed_fixes = true;

    // the last decision, for the harness: the answer, how, and what was not ported ('' all of it); the time of its
    // searches, of its certain lines (and whether a bound cut one: 1 a count, 2 the time)
    struct Trace {
        std::optional<int64_t> o; std::string how, unported, engine, note; int searches = 0, applies = 0;
        double search_s = 0, certain_s = 0, menu_s = 0; int certain_n = 0, certain_cut = 0;   // menu_s: the decision's menu worked out
    } last;
    bool has_plan() const { return bool(plan); }
    bool has_work() const { return work != nullptr; }
    void drop_plan() { plan.reset(); }
    void drop_work() { work = nullptr; work_base = nullptr; diff.reset(); }
    // the memo's Random set to a state (Python's getstate: 624 words, the index, gauss_next); the harness's resync
    void set_memo_rng(const std::vector<uint32_t>& words, uint32_t index, bool has_gauss, double gauss);

private:
    pyobj::Model& m;
    awad::Adapter& ad;
    const awad::Tables& t;
    awdec::Box& box;
    awad::Memo memo;
    bool in_game = false;
    std::string clock_kind;
    // the answer's deadline on a per-decision clock (AlphaWotrBot._deadline: the decide's arrival + the clock less
    // MARGIN_MS); none on other clocks
    std::optional<std::chrono::steady_clock::time_point> answer_deadline;
    awad::PlanPtr plan;
    pyobj::Obj* work = nullptr;        // the engine's state of an ongoing chain
    pyobj::Obj* work_base = nullptr;   // ... and the state it started from
    std::string work_action;
    void* own_rng = nullptr;           // the bot's global stream (no harness)
    uint64_t own_seed = 0;
    using Key = std::tuple<std::string, std::string, std::string>;   // (region slug, nation, unit); "\x01" none
    // diff_answer's board difference
    struct Diff {
        std::vector<std::pair<Key, int64_t>> added, removed;
        std::set<std::pair<std::string, std::string>> chars_to, chars_from, moves_done;
    };
    std::optional<Diff> diff;

    std::string decide(const awjson::Value& m);
    std::pair<std::optional<int64_t>, std::string> decide_(const awjson::Value& dec, const awjson::Value& view, std::optional<double> left);
    std::pair<std::optional<int64_t>, std::string> search_and_map(pyobj::Obj* st, const awjson::Value& dec, const std::string& label,
                                                                  const std::string& flag, std::optional<double> left);
    std::optional<std::pair<std::optional<int64_t>, std::string>> chain(const awjson::Value& dec, std::optional<double> left);
    std::string choose(Menu& mn, const std::set<std::string>& banned, const std::string& label, std::optional<double> left,
                       bool& has_sims, void*& units_menu);
    pyobj::Obj* apply_copy(const void* sn, const void* menu, int64_t index, const std::string& flag);
    bool chain_fits(pyobj::Obj* w, const awjson::Value& dec, const Menu& mn);
    bool chain_over(pyobj::Obj* w, const std::string& kind);
    std::optional<int64_t> diff_answer(const awjson::Value& dec);
    void figure_counts(pyobj::Obj* st, std::vector<std::pair<Key, int64_t>>& out, std::set<std::pair<std::string, std::string>>& chars);
};

}  // namespace awbot
