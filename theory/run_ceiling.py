"""Two checks for theory/02_bouts.md. Writes results/ceiling.csv and results/pruning_by_rank.csv.
(1) Oracle-G ceiling over the referential rate rho and clutter (presence curve scale).
(2) Pruning in developmental order (all more-present objects known) for targets of different rank."""
import itertools, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from single_noun import presence_rates, simulate

V, REPS = 300, 100

def ceiling(job):
    rho, top = job
    p = np.minimum(presence_rates(V, top=top), 0.95)
    base = simulate(rho, p, 49, np.ones(V, bool), reps=REPS, n_max=100_000, seed=7)
    orac = simulate(rho, p, 49, np.ones(V, bool), s=1.0, f=0.0, reps=REPS, n_max=100_000, seed=7)
    return dict(rho=rho, top=top, objects_per_scene=float(p.sum()), p_target=float(p[49]),
                n_base=float(np.median(base)), n_oracle=float(np.median(orac)),
                gain=float(np.median(base / orac)), reached=float(np.isfinite(base).mean()))

def pruning(job):
    rho, rank = job
    p = presence_rates(V)
    t = rank - 1
    known = np.zeros(V, bool); known[:t] = True
    base = simulate(rho, p, t, np.ones(V, bool), reps=REPS, n_max=100_000, seed=11)
    prun = simulate(rho, p, t, ~known, reps=REPS, n_max=100_000, seed=11)
    return dict(rho=rho, target_rank=rank, p_target=float(p[t]), share_of_objects_known=t / (V - 1),
                n_base=float(np.median(base)), n_pruned=float(np.median(prun)),
                gain=float(np.median(base / prun)))

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=10) as ex:
        c = pd.DataFrame(ex.map(ceiling, itertools.product([0.1, 0.05, 0.02, 0.01, 0.005], [0.25, 0.5, 1.0])))
        r = pd.DataFrame(ex.map(pruning, itertools.product([0.05, 0.02], [10, 50, 150, 250])))
    Path("results").mkdir(exist_ok=True)
    c.to_csv("results/ceiling.csv", index=False); r.to_csv("results/pruning_by_rank.csv", index=False)
    print(c.round(3).to_string(index=False)); print(); print(r.round(3).to_string(index=False))
