# theory/

Code and results for the stage-1 analysis. The writeup is the book in `../book/`.

| file | what |
|---|---|
| `single_noun.py` | the model: expected evidence (`T_true`, `C_comp`, `drift`), the bout cues (`bout_cue`, `bout_bank`), the exact-posterior simulator (`simulate`), and checks against the full generative model (`check_drift`, `check_bout_cue`) |
| `run_single_noun.py` | one new noun: pruning and referential cues by referential rate (book chapter 2) → `results/single_noun.csv` |
| `run_bouts.py` | knowledge through the bout cue, default clutter (chapter 3) → `results/bouts.csv`, `results/bouts_runs.csv` |
| `run_ceiling.py` | oracle-gate ceiling by rate and clutter; pruning by target rank (chapters 2–3) → `results/ceiling.csv`, `results/pruning_by_rank.csv` |
| `run_lowrho.py` | low referential rates with high clutter (chapter 3) → `results/lowrho.csv` |

Run from this directory with the repo's venv, for example `../.venv/bin/python run_bouts.py`.
All four scripts together take about a minute on 10 or more cores and are deterministic: rerunning
them reproduces the committed CSVs exactly.
