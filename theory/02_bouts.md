# 2. Bouts: how knowledge reaches the referential variable

Status: 2026-09-22. AI-drafted. Symbols follow [00_model.md](00_model.md). Code: `single_noun.py`
(`bout_cue`, `bout_bank`, `simulate`), `run_bouts.py`, `run_ceiling.py`, `run_lowrho.py`.
Results in `results/`.

## Question

Note 01 showed that with one noun per utterance and referential status independent across
utterances, knowledge can only prune hypotheses. Here referential talk comes in bouts: the gate $G$
is a Markov chain with persistence $\lambda$. A known noun whose object is present, said near an
exposure to the novel noun, is then evidence that the exposure is referential. How much of the
oracle ceiling can knowledge reach this way, and is that ceiling ever an order of magnitude?

## Cues

- **±1 cue:** the number of flagging neighbours, 0 to 2, with the closed-form class-conditional
  probabilities $\alpha_1, \alpha_0$ of 00_model.md section 6.
- **±10 ideal observer:** the exact posterior on $G_t$ given every flag within 10 utterances each side,
  by forward–backward on the gate chain.

## Checks

- **Flag rates vs the full generative model.** I simulated a 3-million-utterance session: gate chain,
  topics, scenes and nouns. Empirical neighbour flag rates around target exposures match
  $\alpha_1, \alpha_0$, for example 0.50 vs 0.48 and 0.110 vs 0.114 at $\rho = 0.05$, $\lambda = 0.8$,
  $K = 0.56$. That includes the prediction that $\alpha_1 = \alpha_0$ at $\lambda = 0$, and the drop in
  $\alpha_1$ under topic persistence $\tau = 0.5$ (0.25 vs 0.25).
- **The ±10 machinery at $W = 1$** reproduces the ±1 cue's distribution exactly.
- **Paired design.** Every condition at a given $\rho$ sees the same scene and gate streams, run by run.
  Gains are medians of per-run ratios over 100 runs.

## Results

**1. Without bouts, knowledge buys nothing.** At $\lambda = 0$ every cue condition gives a gain of exactly
1.00, at every knowledge level and every $\rho$, as derived.

**2. Default clutter (4.1 objects per scene), target object 50.** Gain over no knowledge, with the
target's more frequent nouns known, from pruning plus the ±10 cue:

| $\rho$ | $\lambda$ | K = 0.29 | 0.39 | 0.50 | 0.59 | 0.65 | oracle G |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.8 | 1.05 | 1.24 | 1.37 | 1.64 | 1.51 | 1.93 |
| 0.1 | 0.95 | 1.65 | 1.80 | 1.80 | 1.93 | 1.97 | 1.93 |
| 0.05 | 0.8 | 1.34 | 1.57 | 2.00 | 1.84 | 2.47 | 3.85 |
| 0.05 | 0.95 | 1.89 | 2.61 | 3.32 | 3.26 | 3.64 | 3.85 |
| 0.02 | 0.8 | 1.53 | 2.13 | 2.54 | 2.79 | 3.40 | 6.09 |
| 0.02 | 0.95 | 2.55 | 3.57 | 4.15 | 4.54 | 5.27 | 6.09 |

With long bouts, knowledge reaches most of the ceiling. The ±1 cue reaches roughly half of what
±10 does (`results/bouts.csv`). **So the ceiling is what binds.**

**3. The ceiling grows as referential talk gets rarer and scenes get more cluttered.** Oracle-G gain
for object 50:

| objects per scene | ρ = 0.1 | 0.05 | 0.02 | 0.01 | 0.005 |
|---|---|---|---|---|---|
| 2.1 | 1.8 | 2.8 | 4.6 | 7.9 | 12.9 |
| 4.1 | 1.9 | 2.8 | 5.3 | 10.6 | 18.0 |
| 8.2 | 2.5 | 4.1 | 9.1 | 15.5 | 28.4 |

The no-knowledge cost rises much faster than the oracle cost as $\rho$ falls. At 8.2 objects per scene,
exposures without G go from 47 to about 10,000 between $\rho = 0.1$ and 0.005, while with oracle G they go
from 18 to 360.

**4. At low rates, knowledge reaches an order of magnitude.** High clutter, 8.2 objects per scene;
pruning plus the ±10 cue; median exposures, with the gain in parentheses:

| $\rho$ | $\lambda$ | K = 0 | 0.29 | 0.49 | 0.64 | oracle G |
|---|---|---|---|---|---|---|
| 0.01 | 0.8 | 2,553 | 2,000 (1.1) | 994 (2.5) | 511 (5.2) | 213 (12.7) |
| 0.01 | 0.9 | 2,553 | 1,788 (1.4) | 610 (4.1) | 318 (8.6) | |
| 0.01 | 0.95 | 2,553 | 1,555 (1.6) | 445 (5.6) | 249 (10.1) | |
| 0.005 | 0.8 | 9,307 | 7,643 (1.2) | 2,657 (3.3) | 1,297 (7.2) | 375 (24.2) |
| 0.005 | 0.9 | 9,307 | 5,840 (1.6) | 1,568 (6.6) | 777 (12.5) | |
| 0.005 | 0.95 | 9,307 | 4,529 (1.8) | 951 (9.6) | 544 (18.0) | |

At these rates the gain keeps growing across the whole range of K instead of saturating early, because
the prior on G is low enough that one flag is not decisive.

**5. Pruning in developmental order is minor.** With every more-present object named, the gain from
pruning is 1.0× to 1.1× for targets of rank 10 to 150, and 1.3× to 1.4× for rank 250
(`results/pruning_by_rank.csv`).

**6. Rarer objects are cheaper per exposure at a fixed knowledge level.** At $\rho = 0.02$ with nothing
known, the rank-10 object needs 1,107 exposures and the rank-250 object 200, because a rarely present
object is more distinctive when it co-occurs with its name. Any developmental ladder must therefore
compare each word against its own no-knowledge cost, as the wurwur analysis did, or rarity will pass
for acceleration.

## Comparison with children

Cumulative exposures at acquisition for the 293 nouns in the paper's Fig. 2 (English, pooled M3), by
age of acquisition:

| AoA (months) | 16–20 | 20–24 | 24–28 | 28–32 |
|---|---|---|---|---|
| nouns | 53 | 141 | 78 | 15 |
| median exposures | 1,929 | 747 | 363 | 169 |

Children's nouns span about 1,900 to 170 exposures, an 11× range over the bulk of the distribution. At
$\rho = 0.01$ with 8 objects in view and long bouts, the model's cost for one noun spans 2,553 to 249 as
knowledge grows. The absolute match should not be over-read. The child numbers compare different words
and carry the input-rate anchor's uncertainty, roughly 3× between the low and high anchors, while the
model's numbers compare one word at different knowledge levels. **The point is that the model has a
regime where knowledge-driven acceleration is the size children show.**

## What this says

1. **An order-of-magnitude mechanism exists in this model.** It needs three things at once:
   - **a low rate of real naming.** Naming the object while it is present in about 1% of a noun's uses
     or fewer, consistent with your note that the Gemini rates overstate it;
   - **clutter**, several objects in view;
   - **long referential bouts**, λ of 0.9 or more, so that known nouns nearby inform the gate.
2. **The mechanism is the gate, not the referent.** Pruning is minor. What knowledge buys is the
   ability to tell which uses of a new noun are about something present.
3. **The same conditions make first words expensive.** At ρ = 0.01 a noun learned from nothing needs
   about 2,500 exposures, comparable to children's earliest nouns. That is the ignition question, and it
   can only be answered with earned rather than clamped knowledge.
4. **The mechanism predicts which children and contexts learn faster.** More bout structure (λ), and
   known words that are present when said, should speed learning. Book reading, with high ρ and long
   bouts, is where a learner could start.

## Caveats

- The ideal observer knows ρ, λ, K and φ, and detects whether a known noun's object is present, which is
  easy only because objects are symbols.
- Topics are drawn fresh within a bout (τ = 0). Real bouts repeat the topic, which lowers the flag rate,
  but repeating the novel noun is itself a cue to G that needs no knowledge. Neither is modelled yet.
- One target object and one presence curve per clutter level, with objects present independently.
- Knowledge is clamped. Whether a learner earns it fast enough is the next question.

## Next

1. **Developmental ladder with earned knowledge** at ρ = 0.01, 8 objects, λ = 0.9: does the loop ignite
   from the most frequent nouns, and what does exposures-at-acquisition against acquisition time look
   like, relative to each word's own no-knowledge cost?
2. **Topic persistence and repetition** of the novel noun, τ > 0.
3. **Calibration** from BabyView: per-noun ρ with the name-spoken criterion, bout persistence λ, and
   objects in view. These three numbers decide whether the child's input is in the regime of result 4.
