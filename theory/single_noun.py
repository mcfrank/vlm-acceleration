"""Single-noun analysis: evidence per exposure, ceilings, and what knowledge can buy.

Model (see theory/01_single_noun.md). One novel noun w with true referent o*.
Objects o = 1..V are present in a scene independently with base rates p_o.
Among the exposures to w (episodes in which w is said), a fraction pi are
referential: o* is present and is the topic. Otherwise w is filler and o* is
present only at its base rate.

The learner compares hypotheses h ("w names h") using only w-exposures:
    LR_h(S) = P(S | w said, h) / P(S) = (1 - pi_t) + pi_t * 1[h in S] / p_h,
which is exact when a referential episode forces the topic present and leaves
the other objects at their base rates. pi_t is the learner's probability that
exposure t is referential: pi with no cue, or the posterior given a binary cue.
Presence while unmentioned is neutral: episodes without w are not used.

Knowledge enters in two ways only:
  pruning -- known objects already have names, so by mutual exclusivity they
             are removed from the hypothesis set;
  G-cue   -- a binary cue with P(cue | referential) = s and
             P(cue | not referential) = f sharpens pi_t.
"""
import numpy as np

# ---------- analytic expected log likelihood ratios per exposure ----------

def _xlogy(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, x * np.log(y))

def T_true(pi, p):
    """E[log LR] for the true referent, base rate p, referential fraction pi."""
    pres = pi + (1 - pi) * p
    return _xlogy(pres, 1 - pi + pi / p) + _xlogy((1 - pi) * (1 - p), 1 - pi)

def C_comp(pi, p):
    """E[log LR] for a competitor with base rate p (always <= 0, by Jensen)."""
    return _xlogy(p, 1 - pi + pi / p) + _xlogy(1 - p, 1 - pi)

def cue_states(pi, s, f):
    """[(P(cue state), referential fraction within that state)] for a binary cue."""
    out = []
    for pc_ref, pc_non in [(s, f), (1 - s, 1 - f)]:
        m = pi * pc_ref + (1 - pi) * pc_non
        if m > 0:
            out.append((m, pi * pc_ref / m))
    return out

def drift(pi, p_star, p_comp, s=0.0, f=0.0):
    """Expected per-exposure score advantage of the truth over each competitor."""
    states = cue_states(pi, s, f) if (s or f) else [(1.0, pi)]
    T = sum(m * T_true(r, p_star) for m, r in states)
    C = sum(m * C_comp(r, p_comp) for m, r in states)
    return T - C

def n_pred(pi, p_star, p_comp, s=0.0, f=0.0):
    """Mean-drift prediction: smallest n with sum_h exp(-n D_h) <= 1."""
    D = drift(pi, p_star, np.asarray(p_comp), s, f)
    D = np.where(np.isfinite(D), D, 1e6)
    lo, hi = 0.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sum(np.exp(-mid * D)) <= 1:
            hi = mid
        else:
            lo = mid
    return hi

# ---------- exact-posterior Monte Carlo ----------

def presence_rates(V=300, top=0.5, slope=0.9):
    r = np.arange(1, V + 1)
    return top * r ** (-slope)

def simulate(pi, p, target, admissible, s=0.0, f=0.0, reps=60,
             chunk=1000, n_max=200_000, seed=0):
    """Exposures to w until P(h = o*) first exceeds 0.5. Returns array over reps
    (np.inf if not reached by n_max)."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(admissible)
    tpos = int(np.flatnonzero(idx == target)[0])
    pa = p[idx]
    out = np.full(reps, np.inf)
    with np.errstate(divide="ignore"):
        for r in range(reps):
            score = np.zeros(len(idx))
            n = 0
            while n < n_max:
                G = rng.random(chunk) < pi
                cue = np.where(G, rng.random(chunk) < s, rng.random(chunk) < f)
                if s or f:
                    pc = np.where(cue, pi * s / max(pi * s + (1 - pi) * f, 1e-300),
                                  pi * (1 - s) / max(pi * (1 - s) + (1 - pi) * (1 - f), 1e-300))
                else:
                    pc = np.full(chunk, pi)
                x = rng.random((chunk, len(idx))) < pa
                x[G, tpos] = True
                llr = np.log((1 - pc)[:, None] + pc[:, None] * x / pa[None, :])
                cum = score + np.cumsum(llr, axis=0)
                lse = np.logaddexp.reduce(cum, axis=1)
                post = np.exp(cum[:, tpos] - lse)
                hit = np.flatnonzero(post > 0.5)
                if hit.size:
                    out[r] = n + hit[0] + 1
                    break
                score = cum[-1]
                n += chunk
    return out

def check_drift(pi=0.1, p_star=0.02, p_comp=0.1, s=0.0, f=0.0, n=400_000, seed=1):
    """Monte Carlo mean per-exposure score advantage vs the analytic drift()."""
    rng = np.random.default_rng(seed)
    G = rng.random(n) < pi
    cue = np.where(G, rng.random(n) < s, rng.random(n) < f)
    if s or f:
        pc = np.where(cue, pi * s / (pi * s + (1 - pi) * f), pi * (1 - s) / (pi * (1 - s) + (1 - pi) * (1 - f)))
    else:
        pc = np.full(n, pi)
    xt = G | (rng.random(n) < p_star)
    xc = rng.random(n) < p_comp
    lt = np.log(1 - pc + pc * xt / p_star)
    lc = np.log(1 - pc + pc * xc / p_comp)
    return float(np.mean(lt - lc)), float(drift(pi, p_star, p_comp, s, f))
