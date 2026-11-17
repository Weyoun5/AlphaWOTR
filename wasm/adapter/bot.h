// The AlphaWOTR bot on the module: protocol/wotr_protocol_bot.py's AlphaWotrBot (handle, decide, _decide,
// _search_and_map) and LeagueSearcher.choose at strength 1, over the adapter's port (adapter.h), the plans
// (plans.h) and the decision on the box (driver/decision.h). Phase 4: the engine's chains (the bot's work copy:
// _chain, diff_answer, chain_over) are not ported yet; a question they would answer is reported as such.
#pragma once
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "adapter.h"
#include "decision.h"
#include "plans.h"

namespace awbot {

// one search: the seat's seed, its search index, the search stream's seed, the simulations
struct SearchArgs { uint64_t seat = 0; uint32_t index = 0; uint64_t seed = 0; int64_t sims = 0; };

// AlphaWotrBot._decision_ms(clock): the time per decision the search may use (none: no limit but the simulations)
std::optional<double> decision_ms(const awjson::Value& clock, const std::string& clock_kind);

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
    // LeagueSearcher.budget(kind, left_ms)
    int64_t budget(const std::string& kind, std::optional<double> left_ms) const;

    // the last decision, for the harness: the answer, how, and what was not ported ('' all of it)
    struct Trace { std::optional<int64_t> o; std::string how, unported, engine, note; int searches = 0; } last;
    bool has_plan() const { return bool(plan); }
    bool has_work() const { return work; }
    void drop_plan() { plan.reset(); }
    void set_work(bool w) { work = w; }
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
    awad::PlanPtr plan;
    bool work = false;          // the engine's chain goes on (its copy of the state: not ported)
    uint64_t own_seed = 0;      // the bot's own search seeds (no harness)

    std::string decide(const awjson::Value& m);
    std::pair<std::optional<int64_t>, std::string> decide_(const awjson::Value& dec, const awjson::Value& view, std::optional<double> left);
    std::pair<std::optional<int64_t>, std::string> search_and_map(pyobj::Obj* st, const awjson::Value& dec, const std::string& label,
                                                                  const std::string& flag, std::optional<double> left);
};

}  // namespace awbot
