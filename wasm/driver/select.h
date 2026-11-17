// The root's pick, line for line with the live Python (temperature 0, the default SearchSelectConfig, the
// predecision_profile_v1 reward contract):
//   orchestrator.build_visit_info (native_backend/experiments/native_search_core/own_value_controller_sep13/
//     orchestration/orchestrator.py 649-731): the root view's statistics as the float32 arrays it builds
//   ai.select_root_action (src/ai.py 18489) with _move_to_better and the siege ban (_select_allowed)
//   ai._select_root_action (src/ai.py 18530-18790) at temperature 0: proof dominance, the small-menu rule, the
//     earned-visit default, the terminal-grounded override (conf_margin 0, conf_play off)
// The arithmetic keeps Python's types: float32 where build_visit_info makes np.float32 arrays, float64 where
// _select_root_action converts them, NumPy's pairwise summation for n.sum() and (q*n).sum().
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace awselect {

constexpr int64_t BANNED = -2;                              // ai.BANNED
constexpr int SMALL_MENU = 8;                               // ai._SMALL_MENU
constexpr double SMALL_MIN_GAIN = 0.06, SMALL_MARGIN = 2.0; // ai._SMALL_MIN_GAIN, _SMALL_MARGIN
struct Config {                                             // ai.SearchSelectConfig defaults
    double min_visits = 8, term_frac = 0.95, term_margin = 0.03, completed_q_min_n = 3;
    bool temp_earned = true;
};

// The controller's root view (WotrSearchRoot) plus the verdicts of ai._strictly_better (empty = None).
struct Stats {
    size_t k = 0;
    std::vector<uint64_t> visits, term, forced;
    std::vector<double> value_sum, value_sq, conf_sum, phi_sum, proven_value;
    std::vector<float> priors;
    std::vector<int8_t> proven;
    std::vector<int32_t> side;
    int32_t root_side = 0;
    bool conf_on = false;
    std::vector<int64_t> better;
};

// children_stats of build_visit_info, as _select_root_action reads them (float64 after np.asarray)
struct Arms {
    std::vector<double> p, q, n, tf, pvv, fn, qv;
    std::vector<int8_t> pv;
};

// numpy's pairwise sum (pairwise_sum, PW_BLOCKSIZE 128) for a contiguous float64 array
inline double np_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i + 8 <= n; i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}
// ndarray.sum() of a float64 vector: the reduction starts from the first element and adds the pairwise sum of the rest
inline double np_sum(const std::vector<double>& v) { return v.empty() ? 0.0 : v[0] + np_sum(v.data() + 1, v.size() - 1); }

inline Arms arms(const Stats& s) {
    Arms a;
    const size_t k = s.k;
    for (size_t i = 0; i < k; ++i) {
        const double v = double(s.visits[i]);
        const double qv = s.visits[i] > 0 ? s.value_sum[i] / v : 0.0;
        a.p.push_back(double(s.priors[i]));
        a.n.push_back(double(float(s.visits[i])));                                       // np.float32 visits
        a.q.push_back(double(float(s.side[i] == s.root_side ? qv : -qv)));
        double var = 0.0;
        if (s.visits[i] > 0) { const double m = s.value_sum[i] / v; var = s.value_sq[i] / v - m * m; if (var < 0.0) var = 0.0; }
        a.qv.push_back(double(float(var)));
        a.tf.push_back(double(float(s.visits[i] > 0 ? double(s.term[i]) / v : 0.0)));
        a.pv.push_back(s.proven[i]);
        a.pvv.push_back(double(float(s.proven_value[i])));
        a.fn.push_back(double(int32_t(s.forced[i])));
    }
    return a;
}

inline Arms subset(const Arms& a, const std::vector<size_t>& keep) {
    Arms b;
    for (size_t i : keep) {
        b.p.push_back(a.p[i]); b.q.push_back(a.q[i]); b.n.push_back(a.n[i]); b.tf.push_back(a.tf[i]);
        b.pvv.push_back(a.pvv[i]); b.fn.push_back(a.fn[i]); b.qv.push_back(a.qv[i]); b.pv.push_back(a.pv[i]);
    }
    return b;
}

inline size_t argmax(const std::vector<double>& v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[best] || (std::isnan(v[i]) && !std::isnan(v[best]))) best = i;
    return best;
}

// ai._select_root_action at temperature 0 -> the played index
inline size_t select_root(const Arms& a, const Config& cfg = Config()) {
    const size_t k = a.n.size();
    if (k == 0) return 0;
    const double inf = std::numeric_limits<double>::infinity();
    const double tot = np_sum(a.n);
    std::vector<double> qn(k);
    for (size_t i = 0; i < k; ++i) qn[i] = a.q[i] * a.n[i];
    const double mean_q = tot > 0 ? np_sum(qn) / tot : 0.0;
    std::vector<double> q_hat(k);
    for (size_t i = 0; i < k; ++i) q_hat[i] = a.n[i] >= cfg.completed_q_min_n ? a.q[i] : mean_q;
    // proof dominance under the profile contract: only exact +-1 proofs (_objective_proof_mark)
    std::vector<int8_t> pv(k, 0);
    for (size_t i = 0; i < k; ++i) {
        const int mark = a.pv[i];
        if (mark == 1 || mark == -1) {
            const double bound = mark == 1 ? 1.0 : -1.0, exact = a.pvv[i];
            pv[i] = (std::isfinite(exact) && std::fabs(exact - bound) <= 1e-9) ? int8_t(mark) : 0;
        }
    }
    bool any_win = false, all_lost = true;
    for (size_t i = 0; i < k; ++i) { any_win |= pv[i] == 1; all_lost &= pv[i] == -1; }
    if (any_win) {
        std::vector<double> w(k);
        for (size_t i = 0; i < k; ++i) w[i] = pv[i] == 1 ? q_hat[i] : -inf;
        return argmax(w);
    }
    if (all_lost) return argmax(a.pvv);
    std::vector<bool> alive(k);
    for (size_t i = 0; i < k; ++i) alive[i] = pv[i] != -1;
    // small menu searched full width: the net's own move unless the best mean leads clearly
    if (k >= 2 && k <= size_t(SMALL_MENU) && tot > 0) {
        bool all_forced = true, any_seen = false;
        for (size_t i = 0; i < k; ++i) { all_forced &= a.fn[i] >= a.n[i]; any_seen |= alive[i] && a.n[i] > 0; }
        if (all_forced && any_seen) {
            long play = -1, best = -1;
            for (size_t i = 0; i < k; ++i) {
                if (!(alive[i] && a.n[i] > 0)) continue;
                if (play < 0 || a.p[i] > a.p[play]) play = long(i);                           // max by (p, -i)
                if (best < 0 || a.q[i] > a.q[best] || (a.q[i] == a.q[best] && a.p[i] > a.p[best])) best = long(i);
            }
            const double gain = a.q[best] - a.q[play];
            const double se = std::sqrt(a.qv[best] / a.n[best] + a.qv[play] / a.n[play]);
            if (best != play && gain >= SMALL_MIN_GAIN && gain > SMALL_MARGIN * se) play = best;
            return size_t(play);
        }
    }
    std::vector<bool> elig(k);
    std::vector<double> n_play(k);
    for (size_t i = 0; i < k; ++i) { elig[i] = a.n[i] >= cfg.min_visits && alive[i]; n_play[i] = alive[i] ? a.n[i] : -1.0; }
    if (cfg.temp_earned) {
        bool any_fn = false;
        for (double f : a.fn) any_fn |= f > 0.0;
        if (any_fn) {
            std::vector<double> earned(k);
            bool any_earned = false, any_unforced = false;
            for (size_t i = 0; i < k; ++i) {
                earned[i] = alive[i] ? std::fmax(a.n[i] - a.fn[i], 0.0) : 0.0;
                any_earned |= earned[i] > 0.0;
                any_unforced |= alive[i] && a.fn[i] <= 0.0;
            }
            if (any_earned) for (size_t i = 0; i < k; ++i) n_play[i] = alive[i] ? earned[i] : -1.0;
            else if (any_unforced) for (size_t i = 0; i < k; ++i) n_play[i] = (alive[i] && a.fn[i] <= 0.0) ? a.p[i] : -inf;
        }
    }
    const size_t def = argmax(n_play);
    size_t play = def;
    std::vector<double> w(k);
    bool any_ok = false;
    for (size_t i = 0; i < k; ++i) {
        const bool ok = elig[i] && i != def && (q_hat[i] - q_hat[def]) >= cfg.term_margin &&
                        (a.tf[i] >= cfg.term_frac || a.tf[def] >= cfg.term_frac);
        w[i] = ok ? q_hat[i] : -inf;
        any_ok |= ok;
    }
    if (any_ok) play = argmax(w);
    return play;
}

inline size_t move_to_better(size_t play, const std::vector<int64_t>& better) {
    if (better.empty()) return play;
    std::vector<bool> walked(better.size(), false);
    while (play < better.size() && better[play] >= 0 && !walked[play]) { walked[play] = true; play = size_t(better[play]); }
    return play;
}

// ai.select_root_action: the pick moved on to THE SAME BUT BETTER option; THE SIEGE BAN picks again among the rest
inline size_t pick(const Stats& s, const Config& cfg = Config()) {
    const Arms a = arms(s);
    const size_t play = select_root(a, cfg);
    size_t moved = move_to_better(play, s.better);
    if (!s.better.empty() && moved < s.better.size() && s.better[moved] == BANNED) {
        std::vector<size_t> keep;
        for (size_t i = 0; i < s.better.size(); ++i) if (s.better[i] != BANNED) keep.push_back(i);
        moved = move_to_better(keep[select_root(subset(a, keep), cfg)], s.better);
    }
    return moved;
}

}  // namespace awselect
