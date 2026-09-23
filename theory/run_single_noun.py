"""Grid for theory/01_single_noun.md. Writes results/single_noun.csv."""
import time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from single_noun import presence_rates, simulate, n_pred

V, TARGET = 300, 49                      # target is the 50th most-present object
p = presence_rates(V)
PIS = [0.5, 0.2, 0.1, 0.05, 0.02]

def known_mask(kind, k, seed=0):
    adm = np.ones(V, bool)
    others = np.array([i for i in range(V) if i != TARGET])
    m = int(round(k * (V - 1)))
    if kind == "freq":
        adm[others[:m]] = False           # the most-present objects are named first
    elif kind == "rand":
        adm[np.random.default_rng(seed).choice(others, m, replace=False)] = False
    return adm

CONDS = [("none", "none", 0, 0.0, 0.0),
         ("prune freq 0.5", "freq", 0.5, 0.0, 0.0),
         ("prune freq 0.9", "freq", 0.9, 0.0, 0.0),
         ("prune rand 0.9", "rand", 0.9, 0.0, 0.0),
         ("cue s=.5 f=.02", "none", 0, 0.5, 0.02),
         ("cue s=.9 f=.02", "none", 0, 0.9, 0.02),
         ("oracle G", "none", 0, 1.0, 0.0),
         ("oracle G + prune freq 0.9", "freq", 0.9, 1.0, 0.0)]

rows = []
t0 = time.time()
for pi, (name, kind, k, s, f) in itertools.product(PIS, CONDS):
    adm = known_mask(kind, k)
    comp = p[adm & (np.arange(V) != TARGET)]
    sim = simulate(pi, p, TARGET, adm, s=s, f=f, reps=60, seed=int(1000 * pi))
    rows.append(dict(pi=pi, cond=name, n_sim=float(np.median(sim)),
                     n_sim_q25=float(np.quantile(sim, .25)), n_sim_q75=float(np.quantile(sim, .75)),
                     n_pred=n_pred(pi, p[TARGET], comp, s, f), frac_reached=float(np.isfinite(sim).mean())))
    print(f"{time.time()-t0:6.0f}s  pi={pi:<5} {name:<28} sim={rows[-1]['n_sim']:8.1f}  pred={rows[-1]['n_pred']:8.1f}", flush=True)
Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/single_noun.csv", index=False)
