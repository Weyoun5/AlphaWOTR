// The plans: the referee's follow-up questions of one engine action (protocol/wotr_protocol_bot.py Plan and its
// kinds, lines 1086-1415), the rules for the questions the engine has no decision for (RULES, default_option) and
// region_distance / nearest. A plan answers on the decision's JSON alone.
#pragma once
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "adapter.h"
#include "json.h"

namespace awad {

using Answer = std::optional<int64_t>;   // an option's 'o', or none (the plan has nothing to say)

struct Plan {
    std::vector<std::string> kinds;
    std::string name = "plan";
    bool near = false;                    // an answer was the nearest, not the engine's own
    virtual ~Plan() = default;
    virtual Answer answer(const awjson::Value& dec) = 0;
    bool takes(const std::string& kind) const { for (const auto& k : kinds) if (k == kind) return true; return false; }
};
using PlanPtr = std::shared_ptr<Plan>;

// the steps between two protocol regions on the engine's map (limit + 1: farther or unknown), and the option whose
// `field` region is nearest to `to` (min: the first of the nearest)
int region_distance(const Tables& t, const std::string& a, const std::string& b, int limit = 12);
int region_distance(const Tables& t, const awjson::Value& a, const awjson::Value& b, int limit = 12);   // (None: unknown)
Answer nearest(const Tables& t, const std::vector<const awjson::Value*>& options, const std::string& to, const std::string& field = "to");
Answer nearest(const Tables& t, const std::vector<const awjson::Value*>& options, const awjson::Value& to, const std::string& field = "to");

struct ArmyMovePlan : Plan {
    const Tables& t;
    awjson::Value frm, to;   // protocol slugs (Null: None)
    std::map<std::pair<std::string, std::string>, int64_t> moved;   // (protocol nation, unit) -> count
    std::set<std::string> chars;
    bool everything = false, advance = false, started = false;
    ArmyMovePlan(const Tables& t_, awjson::Value f, awjson::Value to_, std::map<std::pair<std::string, std::string>, int64_t> m,
                 std::set<std::string> c, bool all, bool adv);
    Answer answer(const awjson::Value& dec) override;
};
struct AttackPlan : Plan {
    awjson::Value frm, to;
    AttackPlan(awjson::Value f, awjson::Value to_);
    Answer answer(const awjson::Value& dec) override;
};
struct RecruitPlan : Plan {
    awjson::Value region, nation, unit;   // the fields as compared (Null: absent)
    bool done = false;
    RecruitPlan(awjson::Value r, awjson::Value n, awjson::Value u);
    Answer answer(const awjson::Value& dec) override;
};
struct CompanionMovePlan : Plan {
    std::set<std::string> group;
    awjson::Value frm, to;
    bool started = false;
    CompanionMovePlan(std::vector<std::string> names, awjson::Value f, awjson::Value to_);
    Answer answer(const awjson::Value& dec) override;
};
struct SeparatePlan : Plan {
    std::vector<std::string> left;
    awjson::Value to;
    SeparatePlan(std::vector<std::string> names, awjson::Value to_);
    Answer answer(const awjson::Value& dec) override;
};
struct MinionPlan : Plan {
    const Tables& t;
    std::vector<std::tuple<std::string, awjson::Value, awjson::Value>> flights;   // figure ('nazgul' or a key), from, to
    MinionPlan(const Tables& t_, std::vector<std::tuple<std::string, awjson::Value, awjson::Value>> f);
    Answer answer(const awjson::Value& dec) override;
};
struct CasualtyPlan : Plan {
    std::string key; int64_t k;
    CasualtyPlan(std::string key_, int64_t k_);
    Answer answer(const awjson::Value& dec) override;
};
struct RetreatPlan : Plan {
    std::optional<awjson::Value> to;   // none: stand
    explicit RetreatPlan(std::optional<awjson::Value> to_);
    Answer answer(const awjson::Value& dec) override;
};
struct AnswerPlan : Plan {
    std::string okind;
    AnswerPlan(const std::string& kind, std::string okind_);
    Answer answer(const awjson::Value& dec) override;
};

// casualty_key(dec) (as text) and casualty_pick(dec, reduce_first, plan)
std::string casualty_key(const awjson::Value& dec);
Answer casualty_pick(const awjson::Value& dec, bool reduce_first, CasualtyPlan* plan = nullptr);

// RULES[kind](bot, dec, view): side 0 FP / 1 SH, work: the engine's chain copy exists; nation_in: Names.nation_in's keys
Answer rule(const Tables& t, const std::string& kind, const awjson::Value& dec, const awjson::Value& view, int side, bool work,
            bool* has_rule = nullptr);
// default_option(dec): (o, rule)
std::pair<int64_t, std::string> default_option(const awjson::Value& dec);

}  // namespace awad
