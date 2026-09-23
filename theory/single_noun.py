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
  G-cue   -- a discrete cue with class-conditional pmfs q1 = P(c | referential)
             and q0 = P(c | not referential) sharpens pi_t. A binary cue with
             sensitivity s and false-positive rate f is q1 = [1-s, s], q0 = [1-f, f].
Symbols follow theory/00_model.md.
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

def make_cue(s=0.0, f=0.0, cue=None):
    """Return (q1, q0) arrays, or None for no cue."""
    if cue is not None:
        return np.asarray(cue[0], float), np.asarray(cue[1], float)
    if s or f:
        return np.array([1 - s, s]), np.array([1 - f, f])
    return None

def cue_states(pi, q):
    """[(P(cue value), referential fraction given that value)]."""
    if q is None:
        return [(1.0, pi)]
    q1, q0 = q
    out = []
    for a1, a0 in zip(q1, q0):
        m = pi * a1 + (1 - pi) * a0
        if m > 0:
            out.append((m, pi * a1 / m))
    return out

def drift(pi, p_star, p_comp, s=0.0, f=0.0, cue=None):
    """Expected per-exposure score advantage of the truth over each competitor."""
    states = cue_states(pi, make_cue(s, f, cue))
    T = sum(m * T_true(r, p_star) for m, r in states)
    C = sum(m * C_comp(r, p_comp) for m, r in states)
    return T - C

def n_pred(pi, p_star, p_comp, s=0.0, f=0.0, cue=None):
    """Mean-drift prediction: smallest n with sum_h exp(-n D_h) <= 1."""
    D = drift(pi, p_star, np.asarray(p_comp), s, f, cue)
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

def simulate(pi, p, target, admissible, s=0.0, f=0.0, cue=None, bank=None, reps=60,
             chunk=1000, n_max=200_000, seed=0):
    """Exposures to w until P(h = o*) first exceeds 0.5. Returns array over reps
    (np.inf if not reached by n_max).

    Paired design: the random streams for G, the cue and the scenes are drawn in
    the same order whatever the condition, so runs with the same seed see the same
    data and differ only in what the learner knows. The cue is either discrete
    (s, f or cue=(q1, q0)) or a bank=(pc1, pc0) of posterior referential
    probabilities for referential and non-referential exposures."""
    V = len(p)
    idx = np.flatnonzero(admissible)
    tpos = int(np.flatnonzero(idx == target)[0])
    pa = p[idx]
    out = np.full(reps, np.inf)
    q = make_cue(s, f, cue)
    if q is not None:
        q1, q0 = q
        post_by_c = np.array([pi * a1 / max(pi * a1 + (1 - pi) * a0, 1e-300) for a1, a0 in zip(q1, q0)])
        cq1, cq0 = np.cumsum(q1), np.cumsum(q0)
    with np.errstate(divide="ignore"):
        for r in range(reps):
            rng = np.random.default_rng((seed, r))     # one stream per run, shared across conditions
            score = np.zeros(len(idx))
            n = 0
            while n < n_max:
                G = rng.random(chunk) < pi
                u = rng.random(chunk)
                x = (rng.random((chunk, V)) < p)[:, idx]
                if bank is not None:
                    pc1, pc0 = bank
                    pc = np.where(G, pc1[(u * len(pc1)).astype(int)], pc0[(u * len(pc0)).astype(int)])
                elif q is not None:
                    c = np.where(G, np.searchsorted(cq1, u, side="right"),
                                 np.searchsorted(cq0, u, side="right"))
                    pc = post_by_c[np.minimum(c, len(q1) - 1)]
                else:
                    pc = np.full(chunk, pi)
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


# ---------- the bout cue (route c; theory/00_model.md section 6) ----------

def markov_ab(rho, lam):
    """Transition probabilities a = P(G=1 | G_prev=1), b = P(G=1 | G_prev=0)."""
    return rho + lam * (1 - rho), rho * (1 - lam)

def bout_cue(rho, lam, K, phi, tau=0.0):
    """Class-conditional pmfs of the number of flagging neighbours (0, 1, 2)."""
    a, b = markov_ab(rho, lam)
    a1 = a * (1 - tau) * K + (1 - a) * phi
    a0 = b * K + (1 - b) * phi
    binom = lambda al: np.array([(1 - al) ** 2, 2 * al * (1 - al), al ** 2])
    return binom(a1), binom(a0)

def knowledge_state(p, known):
    """K = token share of known nouns, phi = chance a nonreferential neighbour flags."""
    beta = p / p.sum()
    return float(beta[known].sum()), float((beta[known] * p[known]).sum())

def check_bout_cue(rho=0.05, lam=0.8, tau=0.0, n_known=30, target=49, T=3_000_000, seed=2):
    """Generate one long session from the full model and compare empirical neighbour
    flag rates around exposures to the target noun with alpha_1, alpha_0."""
    rng = np.random.default_rng(seed)
    p = presence_rates()
    V = len(p)
    beta = p / p.sum()
    a, b = markov_ab(rho, lam)
    known = np.zeros(V, bool); known[:n_known] = True
    known[target] = False
    G = np.empty(T, bool); G[0] = rng.random() < rho
    r = rng.random(T)
    for t in range(1, T):
        G[t] = r[t] < (a if G[t - 1] else b)
    fresh = rng.choice(V, size=T, p=beta)
    Z = np.full(T, -1)
    keep = rng.random(T) < tau
    for t in range(T):
        if G[t]:
            Z[t] = Z[t - 1] if (t > 0 and G[t - 1] and keep[t]) else fresh[t]
    noun = np.where(G, Z, rng.choice(V, size=T, p=beta))
    # presence of the noun's own object in its scene
    present = np.where(G, True, rng.random(T) < p[noun])
    flag = known[noun] & present
    idx = np.flatnonzero(noun == target)
    idx = idx[(idx > 0) & (idx < T - 1)]
    nb = np.concatenate([flag[idx - 1], flag[idx + 1]])
    g = np.concatenate([G[idx], G[idx]])
    K, phi = knowledge_state(p, known)
    q1, q0 = bout_cue(rho, lam, K, phi, tau)
    return dict(K=K, phi=phi, exposures=len(idx),
                alpha1_emp=float(nb[g].mean()), alpha1=float(q1[1] / 2 + q1[2]),
                alpha0_emp=float(nb[~g].mean()), alpha0=float(q0[1] / 2 + q0[2]))


def bout_bank(rho, lam, K, phi, W=10, n=200_000, seed=0):
    """Ideal-observer bout cue. For exposures with G_t = 1 and G_t = 0, simulate the
    gate chain W utterances to each side and a flag at each neighbour
    (P(flag | G=1) = K, P(flag | G=0) = phi; tau = 0), then return the exact
    posterior P(G_t = 1 | all 2W flags) from forward-backward. Returns (pc1, pc0),
    posterior samples for referential and non-referential exposures."""
    rng = np.random.default_rng(seed)
    a, b = markov_ab(rho, lam)
    P = np.array([[1 - b, b], [1 - a, a]])          # P[s, s'] = P(G'=s' | G=s)
    emit = np.array([[1 - phi, phi], [1 - K, K]])   # emit[s, flag]
    out = []
    for g in (1, 0):
        side_lik = np.ones((n, 2))                  # P(flags on both sides | G_t = s), s = 0, 1
        for _ in range(2):                          # left and right are i.i.d. given G_t (reversible chain)
            states = np.full(n, g)
            flags = np.empty((n, W), int)
            for k in range(W):
                states = (rng.random(n) < np.where(states == 1, a, b)).astype(int)
                flags[:, k] = rng.random(n) < np.where(states == 1, K, phi)
            beta = np.ones((n, 2))                  # backward pass from the far end
            for k in range(W - 1, -1, -1):
                e = emit[:, flags[:, k]].T          # (n, 2): P(flag_k | state)
                beta = (e * beta) @ P.T             # sum over next state
            side_lik *= beta
        post = rho * side_lik[:, 1] / (rho * side_lik[:, 1] + (1 - rho) * side_lik[:, 0])
        out.append(post)
    return out[0], out[1]
