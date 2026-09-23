Analysis on paper (PLAN.md section 4).

- [00_model.md](00_model.md): the model, with every symbol defined.
- [01_single_noun.md](01_single_noun.md): evidence per exposure for one novel noun, oracle ceilings, and what knowledge can buy when each utterance has one noun. Code: `single_noun.py`, `run_single_noun.py` (run with `../.venv/bin/python run_single_noun.py`, about 10 seconds).
- [02_bouts.md](02_bouts.md): how knowledge reaches the referential variable through bout structure, the size of the ceiling, and the comparison with children's nouns. Code: `run_bouts.py`, `run_ceiling.py`, `run_lowrho.py`, each under a minute on 10 cores.
