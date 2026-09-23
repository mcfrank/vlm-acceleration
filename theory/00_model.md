# 0. The model: definitions and equations

Status: stage-1 specification, 2026-09-22. Every later analysis note uses these symbols.
Stage 1 has one noun per utterance, objects given as symbols, and scenes drawn independently
across episodes. Section 7 lists what is simplified.

## 1. Notation

| symbol | meaning |
|---|---|
| $o \in \{1,\dots,V\}$ | objects |
| $w_o$ | the noun that names object $o$. Nouns and objects are in one-to-one correspondence |
| $t = 1,\dots,N$ | episodes, in order within a session. One episode is one utterance and the scene at its midpoint |
| $S_t \subseteq \{1,\dots,V\}$ | the scene: objects present in episode $t$. $x_{t,o} = \mathbb{1}[o \in S_t]$ |
| $p_o$ | base presence rate of object $o$ |
| $G_t \in \{0,1\}$ | the gate: 1 if utterance $t$ is about something present |
| $\rho$ | stationary referential rate, $P(G_t = 1)$ |
| $\lambda$ | bout persistence, the lag-1 autocorrelation of $G$ |
| $a, b$ | transition probabilities $P(G_t{=}1 \mid G_{t-1}{=}1)$ and $P(G_t{=}1 \mid G_{t-1}{=}0)$ |
| $Z_t \in S_t$ | the topic: the object utterance $t$ is about, defined when $G_t = 1$ |
| $\zeta_o$ | topic propensity: $P(Z_t = o \mid G_t = 1)$ for a fresh topic |
| $\tau$ | topic persistence: probability that a referential utterance keeps the previous referential utterance's topic |
| $u_t$ | the noun said in utterance $t$ |
| $\beta_o$ | background rate: $P(u_t = w_o \mid G_t = 0)$ |
| $\pi_w$ | referential fraction of noun $w$: $P(G_t = 1 \mid u_t = w)$ |
| $\mathcal{K}$ | the set of objects whose names the learner knows |
| $K$ | knowledge: the share of noun tokens that are known, $K = \sum_{o \in \mathcal{K}} \beta_o$ |
| $\mathcal{H}_w$ | hypotheses for a novel noun $w$: objects not already named, $\{1,\dots,V\} \setminus \mathcal{K}$ |
| $c_t$ | a cue to $G_t$ available to the learner |
| $\pi_t$ | the learner's probability that exposure $t$ is referential, $P(G_t = 1 \mid c_t)$ |
| $n$ | exposures to a noun: episodes in which it is said |

## 2. Generative model

**Gate.** $G$ is a stationary two-state Markov chain within a session:

$$P(G_t = 1 \mid G_{t-1} = 1) = a = \rho + \lambda(1 - \rho), \qquad
  P(G_t = 1 \mid G_{t-1} = 0) = b = \rho(1 - \lambda).$$

Then $P(G_t = 1) = \rho$ and $\lambda = a - b$. The mean length of a referential bout is $1/(1 - a)$.
$\lambda = 0$ makes $G$ independent across episodes. A two-state stationary chain is reversible, so
the same transition probabilities hold backward in time.

**Topic.** If $G_t = 1$: when $G_{t-1} = 1$, set $Z_t = Z_{t-1}$ with probability $\tau$. Otherwise draw
a fresh topic, $Z_t \sim \mathrm{Categorical}(\zeta)$.

**Scene.** The topic is present, and every other object is present independently at its base rate:

$$P(S_t \mid G_t = 1, Z_t = h) = \mathbb{1}[h \in S_t] \prod_{o \neq h} p_o^{x_{t,o}} (1 - p_o)^{1 - x_{t,o}}
  = P_0(S_t)\, \frac{\mathbb{1}[h \in S_t]}{p_h},$$

where $P_0(S) = \prod_o p_o^{x_o}(1 - p_o)^{1 - x_o}$ is the scene distribution with no topic. If
$G_t = 0$, then $S_t \sim P_0$.

**Utterance.** If $G_t = 1$, then $u_t = w_{Z_t}$. If $G_t = 0$, then $u_t \sim \mathrm{Categorical}(\beta)$,
independent of the scene.

**Default setting for the simulations.** $V = 300$, with $p_o = 0.5\, o^{-0.9}$ for objects ranked by
presence, which averages $\sum_o p_o = 4.1$ objects per scene. Nouns are named in proportion to how
often their objects are present: $\zeta_o = \beta_o = p_o / \sum_{o'} p_{o'}$. A high-clutter variant
doubles every presence rate, $p_o = \min(0.95,\, 1.0\, o^{-0.9})$, for about 8.2 objects per scene. The
target noun in all analyses so far is object 50, $p^* = 0.015$ by default and $0.030$ in the
high-clutter variant.

## 3. Derived quantities

**Referential fraction of a noun.** By Bayes,

$$\pi_{w_o} = \frac{\rho\, \zeta_o}{\rho\, \zeta_o + (1 - \rho)\, \beta_o}.$$

With $\zeta = \beta$ this is $\pi_w = \rho$ for every noun. The utterance-level rate and the
noun-level rate then coincide, and the two symbols are used interchangeably below.

**Knowledge.** With $\zeta = \beta$, $K = \sum_{o \in \mathcal{K}} \beta_o$ is also the share of referential
utterances whose topic the learner can name.

## 4. The learner

For a novel noun $w$ with true referent $o^*$, the learner keeps a posterior over $h \in \mathcal{H}_w$,
uniform a priori. It uses only the exposures to $w$, the episodes with $u_t = w$. Under hypothesis
$h$, and treating $\pi_w$ as a known property of the noun,

$$P(S_t \mid u_t = w, h) = \pi_t\, P_0(S_t)\, \frac{\mathbb{1}[h \in S_t]}{p_h} + (1 - \pi_t)\, P_0(S_t),$$

so the likelihood ratio against the scene distribution is

$$\boxed{\;\mathrm{LR}_h(S_t) = (1 - \pi_t) + \pi_t\, \frac{\mathbb{1}[h \in S_t]}{p_h}\;}$$

and after $n$ exposures

$$P(h \mid \text{data}) = \frac{\prod_{t=1}^{n} \mathrm{LR}_h(S_t)}{\sum_{h' \in \mathcal{H}_w} \prod_{t=1}^{n} \mathrm{LR}_{h'}(S_t)}, \qquad h \in \mathcal{H}_w.$$

**Acquisition** is the first $n$ at which $P(o^* \mid \text{data}) > 1/2$.

**What is left out, by design.**
- Episodes without $w$ are not used. An object that is present but unmentioned is therefore neutral,
  which removes the penalty that drove the wurwur results.
- The learner does not compare how often $w$ is said with how often $h$ is present. That is the
  "frequency-matching" evidence, and dropping it amounts to treating $\pi_w$ as a single number rather
  than letting it depend on $h$ through $\zeta_h$.

## 5. Expected evidence per exposure

Let $r = \pi / \bigl((1 - \pi)\, p\bigr)$, so that $1 - \pi + \pi/p = (1 - \pi)(1 + r)$. With a constant
$\pi_t = \pi$, the expected log likelihood ratio per exposure is:

$$\text{true referent: } T(\pi, p^*) = \log(1 - \pi) + \bigl(\pi + (1 - \pi) p^*\bigr) \log(1 + r^*),$$
$$\text{competitor: } C(\pi, p') = \log(1 - \pi) + p' \log(1 + r') \;\le 0,$$

since the true referent is present in a fraction $\pi + (1 - \pi)p^*$ of exposures and a competitor in
a fraction $p'$. The truth's expected advantage over a competitor per exposure is the drift
$D(\pi; p^*, p') = T - C$.

- **When $r^* \gg 1$:** $T \approx \pi \log r^*$, linear in $\pi$.
- **When $r^* \ll 1$:** $T \approx \pi^2 (1 - p^*) / (2 p^*)$, quadratic in $\pi$.
- **As $p' \to 1$:** $C \to 0$. A competitor that is always present is never disconfirmed by absence.

**Mean-drift approximation.** The approximate number of exposures to acquisition is the smallest
$n$ with $\sum_{h \neq o^*} e^{-n D_h} \le 1$. It ignores fluctuations and underestimates simulation by
20% to 40%.

## 6. How knowledge enters

**Pruning.** Knowledge of other names changes only $\mathcal{H}_w$. $\mathrm{LR}_h$ depends on the topic
distribution, which depends on the scene and not on the lexicon, so it is unchanged for every
remaining $h$.

**A cue to the gate.** Let the learner observe $c_t$ with class-conditional distributions
$q_1(c) = P(c_t = c \mid G_t = 1)$ and $q_0(c) = P(c_t = c \mid G_t = 0)$. Then

$$\pi_t = \frac{\pi\, q_1(c_t)}{\pi\, q_1(c_t) + (1 - \pi)\, q_0(c_t)},$$

and the expected evidence averages over cue values, each using the referential fraction within it:

$$T_{\text{cue}} = \sum_c \bigl[\pi q_1(c) + (1 - \pi) q_0(c)\bigr]\, T(\pi_c, p^*), \qquad
  C_{\text{cue}} = \sum_c \bigl[\pi q_1(c) + (1 - \pi) q_0(c)\bigr]\, C(\pi_c, p').$$

Where $T$ is convex in $\pi$, splitting exposures into high-$\pi$ and low-$\pi$ classes raises the
average. The value of the cue is this Jensen gap.

**The bout cue (route c).** A neighbouring utterance $t' = t \pm 1$ *flags* when its noun is known and
that noun's object is present in scene $S_{t'}$. The cue is the number of flags,
$c_t \in \{0, 1, 2\}$. Given $G_t$, the two neighbours are independent, because $G_{t-1}$ and $G_{t+1}$
are conditionally independent given $G_t$ in a Markov chain and scenes are drawn independently. So
$c_t \mid G_t = g \sim \mathrm{Binomial}(2, \alpha_g)$, with per-neighbour flag probabilities:

$$\alpha_1 = a\,(1 - \tau)\,K + (1 - a)\,\varphi, \qquad \alpha_0 = b\,K + (1 - b)\,\varphi,$$

$$\varphi = \sum_{o \in \mathcal{K}} \beta_o\, p_o .$$

Here $\varphi$ is the chance that a nonreferential neighbour flags: its noun is known, and that noun's
object happens to be present. A referential neighbour flags when its topic is known and is not the
novel object; its topic is always present. The cue's informativeness is

$$\alpha_1 - \alpha_0 = \lambda\,(K - \varphi) - a\,\tau\,K .$$

- With $\lambda = 0$ and $\tau = 0$, $\alpha_1 = \alpha_0$ and the cue says nothing.
- It grows linearly in $K$, less the chance-presence term $\varphi$, which also grows with $K$ because
  the first objects named are the most present.
- Topic persistence $\tau$ lowers it, because a neighbour that keeps the topic names the novel object
  itself. That repetition is a separate cue to $G_t$ which does not depend on knowledge; it is not
  analysed yet.

**The ideal-observer bout cue.** The learner can instead use every flag within $W$ utterances on
each side. Each utterance $t' \neq t$ emits a flag with $P(\text{flag} \mid G_{t'} = 1) = K$ and
$P(\text{flag} \mid G_{t'} = 0) = \varphi$, taking $\tau = 0$. Utterance $t$ itself carries the novel
noun, so it emits nothing. The learner's referential probability is the exact posterior of the
hidden Markov model,

$$\pi_t = \frac{\rho\, L_1^{\leftarrow} L_1^{\rightarrow}}{\rho\, L_1^{\leftarrow} L_1^{\rightarrow} + (1 - \rho)\, L_0^{\leftarrow} L_0^{\rightarrow}},
\qquad L_g^{\rightarrow} = P(\text{flags at } t+1, \dots, t+W \mid G_t = g),$$

computed by a backward recursion over each side; $L_g^{\leftarrow}$ is the same for the left side by
reversibility. With $W = 1$ this reduces exactly to the binomial cue above. The simulations use $W = 10$.

## 7. Simplifications in stage 1

- One noun per utterance. Route (b), several nouns per utterance, is not modelled yet.
- Objects are symbols with no perceptual noise.
- Objects are present independently, so there are no companions or correlated scenes, and scenes are
  independent across episodes apart from the topic being present.
- Knowledge is set by hand, "clamped", when mapping out efficiency as a function of $K$. The
  developmental ladder, where knowledge is earned, comes later.
