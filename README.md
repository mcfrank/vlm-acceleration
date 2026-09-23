# vlm-acceleration

Can knowledge-dependent latent inference make a word learner more data-efficient as it learns,
by enough to matter, in a world with the referential statistics of children's egocentric input?

Stage 1 is a minimal generative model with an explicit referentiality gate and topic variable,
analysed on paper first and then simulated. A BabyView VLM experiment comes later and is designed
from the model's predictions.

**Status:** planning only. Read [PLAN.md](PLAN.md). No code yet.

## Layout

| path | what |
|---|---|
| [`PLAN.md`](PLAN.md) | the design and the decisions made so far |
| `theory/` | analysis on paper: evidence per exposure, ceilings, ignition |
| `sim/` | stage-1 simulations of the symbolic model (batch developmental ladder) |
| `babyview/` | calibration statistics from BabyView, and later the VLM experiment |

BabyView is human-subjects data. Frames, transcripts and per-utterance annotations stay on the
cluster. Only aggregate statistics are committed here.

Related repos on this machine: `~/Projects/acceleration` (the child vs LM acceleration paper),
`~/Projects/wurwur-acceleration` (the FGT 2009 simulations), `~/Projects/vlm-headcam`
(BabyView word learning).
