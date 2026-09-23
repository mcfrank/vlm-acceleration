"""Check for theory/02_bouts.md: does knowledge, through the ideal-observer bout cue
(with pruning of the known objects), reach the oracle ceiling at low referential rates
and high clutter?
Writes results/lowrho.csv."""
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from single_noun import presence_rates, simulate, bout_bank, knowledge_state

V, TARGET, REPS, W = 300, 49, 100, 10
p = np.minimum(presence_rates(V, top=1.0), 0.95)      # about 8 objects per scene

def run(job):
    rho, mode, lam, j = job
    known = np.zeros(V, bool); known[:j] = True; known[TARGET] = False
    K, phi = knowledge_state(p, known)
    seed = int(rho * 10000)
    kw = {"none": {}, "oracle G": dict(s=1.0, f=0.0)}.get(mode)
    if mode == "cue":
        kw = dict(bank=bout_bank(rho, lam, K, phi, W=W, seed=j))
    sim = simulate(rho, p, TARGET, ~known, reps=REPS, n_max=100_000, seed=seed, **kw)
    return [dict(rho=rho, mode=mode, lam=lam, j=j, K=K, phi=phi, run=r, n=float(x)) for r, x in enumerate(sim)]

if __name__ == "__main__":
    js = [(r, m, np.nan, 0) for r in (0.01, 0.005) for m in ("none", "oracle G")]
    js += [(r, "prune + cue", lam, j) for r, lam, j in itertools.product([0.01, 0.005], [0.8, 0.9, 0.95], [5, 20, 49])]
    js = [(r, "cue" if m == "prune + cue" else m, lam, j) for r, m, lam, j in js]
    with ProcessPoolExecutor(max_workers=10) as ex:
        runs = pd.DataFrame([row for rows in ex.map(run, js) for row in rows])
    base = runs[runs["mode"] == "none"].set_index(["rho", "run"]).n
    runs["gain"] = base.loc[list(zip(runs.rho, runs.run))].values / runs.n
    summ = runs.groupby(["rho", "mode", "lam", "j", "K"], dropna=False).agg(n_median=("n", "median"), gain=("gain", "median")).reset_index()
    Path("results").mkdir(exist_ok=True); summ.to_csv("results/lowrho.csv", index=False)
    print(summ.round(3).to_string(index=False))
