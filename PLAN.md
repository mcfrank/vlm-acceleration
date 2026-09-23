# Plan: a minimal model of knowledge-dependent acceleration

Planning draft for discussion (2026-09-22). AI-drafted under M. C. Frank's direction. No code yet.

## 1. What we already know

| source | what it shows | size |
|---|---|---|
| wurwur (FGT 2009) simulations | knowledge speeds new words through referent inference, but by relieving an unmentioned-object penalty | 2.4× at 4 objects per scene |
| Steven's LM dose assay | later CHILDES-ladder rungs learn more from a fixed dose of novel-word evidence | about 2× from 0.5M to 24M words; direction robust, magnitude sensitive to update size |
| BabyView oracle filter | training on referentially aligned pairs only | 20× to 34× raw-data advantage |
| BabyView self-bootstrap | the learner's own alignment estimate vs Gemini | ρ ≈ 0.1, against a gate threshold of 0.3 to 0.5 |
| children, paper Fig. 2 | exposures at acquisition fall steeply with age of acquisition | nouns: slope ≈ −4.8 on log–log axes |

A single stage of knowledge-dependent inference has now given about 2× in two different learners.
The question for this repo is whether any mechanism of this kind can reach an order of magnitude,
and what the input has to look like for it to.

## 2. Evaluation standards (decided)

Primary readouts, all on a log-odds or evidence scale with the acquisition threshold fixed in advance:

1. **Clamped gain.** Efficiency with a fraction k of other words given, over efficiency with none, same data. The causal measure.
2. **Dose slope.** Steven's assay: give learners at knowledge K a fixed dose of novel-word episodes and measure learning per unit dose. Puts the model in the same units as the LM results.
3. **Exposures at acquisition vs time of acquisition** on a developmental ladder (the elasticity ε), the analog of paper Fig. 2.
4. **Mechanism chain.** Posterior accuracy on the gate and topic variables as a function of K, and realized savings per word as a function of K at acquisition.

κ is not used as a diagnostic.

## 3. The model

**World.**
- V objects, each with one true name, Zipfian frequencies. Function words.
- A fixed background language model P_bg over the vocabulary, independent of the learner's lexicon. Object names appear in P_bg at their corpus rate, whether or not the object is present.
- Episode t has a scene S_t of c_t objects, with c drawn from the BabyView detection distribution.
- **Gate** G_t ∈ {0,1}: is the utterance about something present. G follows a two-state Markov chain with stationary rate ρ and persistence λ, so referential moments come in bouts. λ = 0 is the iid case.
- **Topic** Z_t ∈ S_t when G_t = 1, persisting within a bout with probability τ while the object stays present.
- **Utterance.** If G = 1: the name of Z_t plus m − 1 other tokens from P_bg, optionally with co-referring names of other present objects. If G = 0: all tokens from P_bg.

**Learner.** A posterior over the lexicon. Per episode it infers (G_t, Z_t) from the scene, the utterance, neighbouring episodes and its current lexicon, then updates.

**Design constraint from wurwur.** A present but unmentioned object is never evidence against its name. It is simply not the topic, or the episode is not referential. Knowledge can help only through positive routes:

- **(a) Mutual exclusivity on Z.** A known object present whose name was not said is ruled out as the topic, so the candidate set shrinks from c to the unknown objects.
- **(b) G from language.** A known word whose referent is present is evidence for G = 1. Known words whose referents are absent are evidence for G = 0.
- **(c) G through bouts.** With λ > 0, route (b) in neighbouring utterances reaches the utterance that carries the novel word.

## 4. Analysis on paper first

- **A1. Evidence per exposure.** Expected log-likelihood ratio per exposure for a novel word's true edge as a function of (ρ, c, k, λ, τ, m). This generalizes the two-episode arithmetic in the wurwur book.
- **A2. The ceiling.** The gain from oracle G and Z over K = 0. Conjecture: roughly (1/ρ)·g(c) for small ρ, because an unlabeled mixture component of weight ρ carries Fisher information of order ρ², against ρ when labeled. Check the prediction against the BabyView oracle filter, 20× to 34× at ρ ≈ 0.07 to 0.10.
- **A3. How much of the ceiling knowledge can reach.** Prediction to test: with λ = 0 and one content word per utterance, route (b) says nothing about the novel word's own episode. Knowledge then acts only through Z, bounded by about c, which is the wurwur regime. Gains of an order of magnitude need λ > 0 or m > 1. If this holds, discourse structure becomes the key empirical quantity.
- **A4. Ignition as a horizon question.** Without negative evidence every word is learnable in the limit. The question becomes whether the first words are learnable within a child-scale horizon, for example one year of input at about 700 utterances per hour and 4,000 waking hours. This replaces wurwur's hard cliff at 6 objects.

## 5. Simulations

**Learners.**
- A batch learner with blocked Gibbs over the lexicon, reusing the scoring and ladder machinery in `~/Projects/wurwur-acceleration/src/wacc`.
- An incremental soft-EM learner: per-word posteriors, each episode seen once. This is the literal η(K) case and the closest analog to VLM training.

**Controls.** Oracle G, oracle Z, both; no inference, pure co-occurrence counting; clamped K at 0, 0.25, 0.5, 0.75 and 1; frozen inference, with the E-step's lexicon fixed at an early checkpoint.

**Regimes.** ρ ∈ {1, 0.5, 0.1, 0.07}; c from 2 up to the BabyView distribution; λ ∈ {0, 0.5, 0.8, 0.95}; m ∈ {1, 2, 3}, with and without co-reference.

**Must-pass conditions.** The gain has to survive m = 2 and ρ = 0.1, the two regimes that broke wurwur.

## 6. Calibration from BabyView

Aggregate statistics only leave the cluster. From the vlm-headcam Gemini annotations and detections:

- ρ: already known, 10% aligned and 7% with the referent spoken.
- Scene size distribution, from detections.
- Content tokens per utterance.
- **λ: autocorrelation of Gemini alignment across consecutive utterances.** New, cheap, and the most consequential number for A3.
- τ: persistence of the Gemini referent across consecutive aligned utterances.
- How often a noun is said while its object is present but is not the referent, and while its object is absent.

## 7. Stage 2, sketch only

- **Compounding.** The learner trains its own G-detector on its confident episodes, using features other than known words, such as utterance form and discourse position, so that it generalizes to episodes with no known word. Does a second stage multiply the first? This is the connection to Korchinski, Favero and Wyart (2026). The difference is that their lowest latent is recoverable from the data alone, while G is not recoverable at K = 0.
- **Perceptual noise.** Objects as noisy percepts rather than symbols.

## 8. Bridge to a BabyView experiment, later

The model should deliver three things: the ρ that the learner's own G-inference must reach as a function of K; whether lexical knowledge alone can supply it at the measured λ; and how large an external spark, gems or a social cue, is needed to start the loop. The VLM experiment then tests a rung-dependent referentiality readout with those predictions stated in advance.

## 9. Open decisions

1. Batch or incremental learner as the primary model.
2. Objects as symbols in stage 1. Proposed: yes.
3. Whether to mirror Steven's dose assay exactly.
4. Scope: model only, or also the VLM experiment later.
5. Target words: object names only, to match the noun panel of paper Fig. 2.
