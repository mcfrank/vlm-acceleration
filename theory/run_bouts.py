"""Grid for theory/02_bouts.md: exposures to acquire the target noun as a function of
knowledge, when knowledge acts by pruning, through the bout cue, or both. Paired
design: every condition at a given rho sees the same data streams.
Writes results/bouts_runs.csv (one row per run) and results/bouts.csv (summary)."""
import itertools, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from single_noun import presence_rates, simulate, bout_cue, bout_bank, knowledge_state

V, TARGET, REPS = 300, 49, 100
p = presence_rates(V)
RHOS = [0.1, 0.05, 0.02]
JS = [0, 5, 10, 20, 35, 49]             # the j most-present objects other than the target are known
LAMS = [0.0, 0.5, 0.8, 0.95]
W = 10                                  # ideal-observer window: +-W utterances

def known_set(j):
    k = np.zeros(V, bool); k[:j] = True; k[TARGET] = False
    return k

def jobs():
    for rho in RHOS:
        yield (rho, 0, "none", None)
        yield (rho, 0, "oracle G", None)
        yield (rho, 49, "oracle G + prune", None)
        for j in JS[1:]:
            yield (rho, j, "prune", None)
            for lam in LAMS:
                yield (rho, j, "cue +-1", lam)
                yield (rho, j, f"cue +-{W}", lam)
                yield (rho, j, f"prune + cue +-{W}", lam)

def run(job):
    rho, j, mode, lam = job
    known = known_set(j)
    K, phi = knowledge_state(p, known)
    adm = ~known if "prune" in mode else np.ones(V, bool)
    kw = {}
    if mode.startswith("oracle"):
        kw = dict(s=1.0, f=0.0)
    elif mode.endswith(f"+-{W}"):
        kw = dict(bank=bout_bank(rho, lam, K, phi, W=W, seed=j))
    elif mode.endswith("+-1"):
        kw = dict(cue=bout_cue(rho, lam, K, phi))
    sim = simulate(rho, p, TARGET, adm, reps=REPS, n_max=50_000, seed=int(rho * 1000), **kw)
    return [dict(rho=rho, j=j, K=K, phi=phi, mode=mode, lam=lam if lam is not None else np.nan,
                 run=r, n=float(x)) for r, x in enumerate(sim)]

if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        runs = pd.DataFrame([row for rows in ex.map(run, list(jobs())) for row in rows])
    Path("results").mkdir(exist_ok=True)
    runs.to_csv("results/bouts_runs.csv", index=False)
    base = runs[runs["mode"] == "none"].set_index(["rho", "run"]).n
    runs["gain"] = base.loc[list(zip(runs.rho, runs.run))].values / runs.n
    summ = (runs.groupby(["rho", "j", "K", "phi", "mode", "lam"], dropna=False)
                .agg(n_median=("n", "median"), gain_median=("gain", "median"),
                     gain_q25=("gain", lambda g: g.quantile(.25)), gain_q75=("gain", lambda g: g.quantile(.75)),
                     reached=("n", lambda x: np.isfinite(x).mean()))
                .reset_index())
    summ.to_csv("results/bouts.csv", index=False)
    print(f"{len(summ)} cells, {len(runs)} runs in {time.time()-t0:.0f}s; min reached {summ.reached.min()}")
