# vlm-acceleration

Can knowledge-dependent inference make a word learner more data-efficient as it learns, by enough
to matter, in a world with the referential statistics of children's egocentric input?

**Read the book:** the writeup is a Quarto book in [`book/`](book/), published at
<https://mcfrank.quarto.pub/knowledge-dependent-acceleration-in-word-learning>. Start with its overview, which summarizes the findings and the handoff status.

## Status

- **Done:** stage-1 theory with clamped knowledge. A minimal generative model with an explicit
  referential gate, topic and bout structure; exact-posterior simulations; analytic formulas checked
  against simulation.
- **Next:** a developmental ladder with earned knowledge; topic persistence; several nouns per
  utterance; calibration from BabyView; then a BabyView vision–language model experiment designed from
  the model's predictions. Book chapter 4 has the plan.

## Layout

| path | what |
|---|---|
| [`book/`](book/) | the writeup (Quarto book). `./render.sh` renders it; chapters compute every table and figure from `theory/results/` |
| [`theory/`](theory/) | model code, the four simulation scripts, and their results (CSV) |
| `sim/` | developmental-ladder simulations with earned knowledge (not started) |
| `babyview/` | BabyView calibration and, later, the model experiment (not started) |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name vlm-acceleration
cd theory && for s in run_single_noun run_bouts run_ceiling run_lowrho; do ../.venv/bin/python $s.py; done
cd ../book && ./render.sh
```

The simulations take about a minute in total and reproduce the committed CSVs exactly. To publish
an update to the same URL, render and then run `quarto publish quarto-pub --no-render` from `book/`;
`book/_publish.yml` records the target.

## Data

BabyView is human-subjects data. Frames, transcripts and per-utterance annotations stay on the
cluster; only aggregate statistics may be committed here. The book is public, so anything added to
it must be cleared for public release.

## Related work in the lab

- The developmental paper: Frank, *Children, but not language models, show accelerating returns in
  word learning* (arXiv, 2026), source of the children's exposure estimates.
- Simulations of the Frank, Goodman & Tenenbaum (2009) word-learning model, which motivated the
  design here.
- The BabyView vision–language word-learning project, source of the referential-alignment estimates.

AI-drafted (Claude Code) under M. C. Frank's direction; see the caution boxes in the book.
