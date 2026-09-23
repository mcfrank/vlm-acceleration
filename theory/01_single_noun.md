# 1. One novel noun: evidence per exposure, and what knowledge can buy

Status: first pass, 2026-09-22. AI-drafted. Code: `single_noun.py` (model and simulator),
`run_single_noun.py` (grid). Results: `results/single_noun.csv`.

## Model

One novel noun w with true referent o*. Objects o = 1..V are present in a scene independently
with base rates p_o. Among the exposures to w, a fraction π are **referential**: o* is present
and is the topic. The rest are filler, and o* is present only at its base rate.

A referential episode forces the topic present and leaves the other objects at their base rates,
so P(S | referential, topic h) = P(S)·1[h∈S]/p_h. The learner compares hypotheses h, "w names h",
using only the exposures to w:

    LR_h(S) = (1 − π_t) + π_t · 1[h ∈ S] / p_h

where π_t is the learner's probability that exposure t is referential: π with no information,
or the posterior given a cue. Episodes in which w is not said are not used, so an object that is
present but unmentioned is neutral. This is the design constraint from the wurwur simulations. The
learner also ignores how often w is said relative to how often h is present.

## A1. Evidence per exposure

Let r = π / ((1 − π) p). The expected log likelihood ratio per exposure is

    true referent:   T(π, p*) = log(1 − π) + (π + (1 − π) p*) · log(1 + r*)
    competitor:      C(π, p') = log(1 − π) + p' · log(1 + r')          (always ≤ 0)

and the truth gains D = T − C per exposure on a competitor with base rate p'. These match Monte
Carlo to within 2% (`check_drift`).

**Two regimes.** When r ≫ 1, T ≈ π log r, linear in π. When r ≪ 1, that is when the referential
rate is below the competitors' base rates, T ≈ π²(1 − p*)/(2p*), **quadratic in π**. BabyView
nouns are likely in or near the quadratic regime.

**The hardest competitors are the most-present objects.** As p' → 1, C → 0: an object that is
always there is never disconfirmed by its absence. Rare competitors lose about −log(1 − π) per
exposure on top of that.

## A3. What knowledge can do when each utterance has one noun and referential status is independent across episodes

The probability that w is said in scene S under hypothesis h depends on the topic distribution,
which depends on the scene and not on the lexicon. So knowing other objects' names leaves
LR_h(S) unchanged for every hypothesis that remains admissible. Episodes in which a known noun is
said carry no information about w. **Knowledge can act only by pruning**: by mutual exclusivity,
objects that already have names leave the hypothesis set.

For knowledge to reach the referential variable G_t, the learner needs a known word that bears on
exposure t. That requires another noun in the same utterance (route b) or a known noun in the same
referential bout (route c). Neither exists in this world. The G-cue conditions below are therefore
ceilings on what those routes could buy, not predictions.

**How a cue helps.** T and C are computed within each cue state, using the referential fraction in
that state, and then averaged. Where T is quadratic in π, splitting exposures into high-π and
low-π states raises the average. The value of G information is this Jensen gap, so it grows as π
falls.

## Simulations

Exact posterior over 300 objects, with presence falling as rank^−0.9 from 0.5, averaging 4.1 objects
per scene. The target is the 50th most present object, p* = 0.015. Acquisition is the first
exposure at which P(o*) exceeds 0.5. Cells are medians over 60 runs and carry roughly ±15% noise.
Pruning removes a fraction k of the other objects, either the most present first, which is the
order in which they would be named, or at random. The cue has sensitivity s and false-positive
rate f.

Exposures to acquire the noun:

| condition | π = 0.5 | 0.2 | 0.1 | 0.05 | 0.02 |
|---|---|---|---|---|---|
| no knowledge | 3.5 | 13.5 | 35.5 | 101.5 | 430.5 |
| prune most-present 50% | 4.0 | 12.0 | 30.0 | 72.0 | 382.0 |
| prune most-present 90% | 2.0 | 8.0 | 22.0 | 39.5 | 183.0 |
| prune random 90% | 2.0 | 9.0 | 24.5 | 55.0 | 246.5 |
| cue s = 0.5, f = 0.02 | 3.5 | 9.5 | 21.5 | 69.5 | 157.0 |
| cue s = 0.9, f = 0.02 | 3.0 | 10.0 | 18.5 | 34.0 | 85.0 |
| oracle G | 3.0 | 8.0 | 14.5 | 33.5 | 75.0 |
| oracle G + prune most-present 90% | 1.0 | 5.0 | 6.5 | 19.0 | 31.0 |

Gain over no knowledge:

| condition | π = 0.2 | 0.1 | 0.05 | 0.02 |
|---|---|---|---|---|
| prune most-present 90% | 1.7× | 1.6× | 2.6× | 2.4× |
| prune random 90% | 1.5× | 1.5× | 1.8× | 1.8× |
| cue s = 0.9, f = 0.02 | 1.4× | 1.9× | 3.0× | 5.1× |
| oracle G | 1.7× | 2.4× | 3.0× | 5.7× |
| oracle G + prune most-present 90% | 2.7× | 5.5× | 5.3× | 13.9× |

## What this says

1. **A noun's cost rises steeply as π falls.** From π = 0.1 to 0.05 to 0.02, exposures go
   36 → 102 → 430, roughly π^−1.5 and steepening toward the quadratic limit.
2. **Pruning alone buys about 2×.** Nine-tenths of the lexicon known gives 1.6× to 2.6×. Pruning in
   naming order beats random pruning because the first objects named are the most present, and the
   most present are the hardest competitors. This is a third instance of "one stage buys about 2×",
   after the wurwur simulations (2.4×) and Steven's LM assay (about 2×).
3. **Referential information is worth more the lower π is.** Oracle G gives 2.4× at π = 0.1 and 5.7×
   at 0.02. Combined with pruning it reaches 14× at π = 0.02.
4. **The cue does not have to be clean.** At π = 0.02, a cue with s = 0.9 and f = 0.02 captures most
   of the oracle gain (5.1× vs 5.7×), even though only about half the exposures it flags are
   referential.
5. **An order of magnitude needs both conditions at once:** a low referential rate, and knowledge that
   carries information about G. With one noun per utterance and no bout structure, knowledge carries
   none, and the ceiling is pruning's 2×.

**Correction to PLAN.md.** Section 4 conjectured a ceiling of roughly (1/ρ)·g(c). The oracle-G gain
does grow as π falls, but much more slowly than 1/π: 5.7× at π = 0.02, not 50×. Absence eliminates a
competitor only once G is known, and competitors that co-occur with o* still have to be outlasted,
so clutter sets the ceiling. The BabyView oracle filter's 20× to 34× is not directly comparable.
That filter also supplies the referent and which word names it, and the VLM is not this learner.

**Mean-drift approximation.** Predicting exposures from the expected drift alone, as the smallest n with
Σ_h exp(−n D_h) ≤ 1, underestimates simulation by 20% to 40% because it ignores fluctuations. It
gets the ordering right, and it is undefined for oracle G, where absence eliminates. It is a guide
only.

## Caveats

- Objects are present independently. There are no companions and no correlated scenes.
- One presence curve, one target rank, and the 0.5 first-crossing criterion.
- The learner knows π and ignores frequency-matching evidence.
- Medians of 60 runs, so single cells are noisy. The cue at s = 0.5 is non-monotone across π.

## Next

1. **Route (c): map knowledge to a cue.** In a referential bout of B utterances, a known noun
   elsewhere in the bout whose referent is present flags G. That gives s ≈ 1 − E[(1 − K)^(B−1)],
   with K the token-weighted share of nouns known. f comes from known nouns' referents being present
   by chance in nonreferential stretches. Plugging s(K) and f(K) into the simulator gives predicted
   exposures as a function of K, which is the acceleration curve. Route (b), several nouns per
   utterance, is the same exercise within an utterance.
2. **A4: ignition.** Compare exposures needed at K = 0 with how often a child hears each noun in a
   year, to find which nouns can be learned first at a given π.
3. **Calibration** now hinges on per-noun π, measured as name spoken while the object is present,
   and on bout persistence. See PLAN.md section 6.
