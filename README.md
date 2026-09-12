# Where Is It? — Bayesian Roomba Localization with a Mobile Detector

**Tracking a sensorless space roomba from a distance using recursive Bayesian filtering, an exponential-decay proximity detector, and decision-theoretic action selection.**


> Companion project to [**Belief-Space-Localization**](../) (see project 1) — that project taught the roomba to localize *itself* by acting. This project flips the problem: **you** are the *LocatorBot*, and you must figure out where the roomba is using probability and a noisy detector, without ever seeing it directly.

---

## Scenario

The roomba's sensors are fried again — it can't localize itself, and now it's wandering the ship blindly while you, the **LocatorBot**, try to track it down by radio. You have no camera, no map overlay of its position — only a **beep / no-beep detector** whose accuracy decays with distance, and the ability to issue it movement commands. The task: figure out which cell it's in using as few total actions (moves + detector activations) as possible.


---

## 1. The Problem

This time there are **two agents**: the space roomba (still sensorless, still blind) and *you*, the LocatorBot, who can see nothing directly but can:

1. **Issue a movement command** to the roomba (`N/E/S/W`) — it attempts to move, and may bounce off a wall.
2. **Run a detector** — it emits a probabilistic beep depending on how far away the roomba actually is:

$$
P(\text{beep} \mid \text{Locator in } a,\ \text{Roomba in } b) = e^{-\alpha(\text{dist}(a,b) - 1)}, \qquad
P(\neg\text{beep} \mid a, b) = 1 - e^{-\alpha(\text{dist}(a,b)-1)}
$$

where $\alpha > 0$ is the detector's sensitivity — small $\alpha$ means a long-range but noisy sensor (frequent false positives), large $\alpha$ means a short-range but crisp one.

Unlike Project 1 (where the goal was to collapse a *set* of possible locations to a singleton via pure logical elimination), here the belief is a full **probability distribution** over cells, updated after every move and every detector activation. The task ends the moment either (a) the belief becomes certain, or (b) the roomba physically walks into the LocatorBot's cell. The objective being minimized is **total actions = moves + sensor activations** — so a strategy that senses constantly but rarely needs to move can still lose to one that thinks harder before acting.

---

## 2. The Math — Belief Update Equations

### Motion update (prediction step)

After issuing move command $u_t$, the prior belief is convolved with the roomba's motion model:

$$
b_t^{-}(b) = \sum_{a \in A} P(x_t = b \mid x_{t-1} = a,\ u_t)\, b_{t-1}(a)
$$

with a simple wall-aware transition model — the intended move succeeds most of the time and bounces the rest:

$$
P(x_t = b \mid x_{t-1}=a, u_t) =
\begin{cases}
0.8 & \text{move succeeds, } b = \text{neighbor}(a, u_t) \text{ is open} \\
0.2 & \text{move fails/blocked, } b = a \\
0 & \text{otherwise}
\end{cases}
$$

If the intended move happens to land the roomba on the LocatorBot's own cell, the same equation applies — the run simply terminates successfully the moment $x_t = L$.

### Sensor update (correction step)

After activating the detector and observing outcome $z_t$, Bayes' rule folds the reading into the belief:

$$
b_t(b) = \eta\, P(z_t \mid x_t = b)\, b_t^{-}(b)
$$

where $\eta$ is a normalizing constant and the likelihood is the exponential-decay detector model:

$$
P(z_t \mid x_t = b) =
\begin{cases}
e^{-\alpha(d(b, L) - 1)} & z_t = \text{beep} \\
1 - e^{-\alpha(d(b, L)-1)} & z_t = \neg\text{beep}
\end{cases}
$$

Every strategy below is built on top of exactly these two update rules — what differs between them is **when to sense, when to move, and where.**

---

## 3. Approach

### Baseline Strategy 1 — reuse Project 1

Runs Project 1's optimal belief-space A\* search to drive the roomba into a fully known cell purely through forced-wall-bounce logic, with **no sensing at all**. It's a useful floor to compare against — it shows what pure deduction costs versus what probabilistic sensing buys you — but it never uses the detector, and its "moves" number isn't directly comparable to a moves + senses total.

### Baseline Strategy 2 — greedy MAP chasing (`baseline_strategy_2.py`)

The given strategy: at every step, run the detector, do one Bayesian update, then issue a command that moves the *most probable* candidate cell one step closer to the LocatorBot. It's simple and does converge, but it's entirely myopic — it senses on every single iteration regardless of whether the current belief is already informative, and it always chases the current MAP estimate even when that estimate is barely more probable than its neighbors. This wastes actions when α is small (the detector is barely informative per-reading) and can oscillate when the belief is genuinely ambiguous.

### Developed Strategy — decision-theoretic Bayesian filtering (`developed_strategy.py`)

The Developed Strategy keeps Baseline 2's recursive Bayesian core but wraps it in a **decision-theoretic layer** that reasons about the *value* of information rather than just chasing probability mass. At each step it:

1. **Predicts** — propagates the belief forward through the transition model (Eq. 1–2), with an exploration bias: if the current MAP certainty is low (< 0.6), it issues a random move to actively *disambiguate* the belief rather than blindly converging on a possibly-wrong peak.
2. **Corrects** — folds in the new detector reading via Bayes' rule (Eq. 3–5).
3. **Scores the state** with a decomposable utility function:

$$
U_t = -\big(w_m \cdot \text{moves} + w_s \cdot \text{senses} + w_u \cdot (1 - \text{belief certainty})\big)
$$

Rather than greedily moving toward whatever cell currently looks most probable, the strategy is explicitly accounting for the *cost* of every move and sense against the *value* of reduced uncertainty. This design is grounded in classical utility theory — completeness/transitivity (actions can always be compared by expected utility), continuity (small probability shifts produce proportional preference shifts), and monotonicity/decomposability (each unit of certainty gained adds utility additively) — which is what justifies treating $U_t$ as a valid basis for action selection rather than an ad-hoc score.

**Empirical result:** at low $\alpha$ (weak, noisy detector), both Baseline 2 and the Developed Strategy hit the action cap (~400) because the sensor simply isn't informative enough to converge quickly. But once $\alpha \geq 0.5$ (a moderately sharp detector), the Developed Strategy converges in **36 total actions vs. Baseline 2's 400**, with a normalized utility of **0.514** — because it recognizes when sensor readings have become trustworthy and exploits them instead of continuing to move reflexively.

### Mobile Developed Strategy — a LocatorBot that can also move (`mobile_developed_strategy.py`)

The final extension allows the *LocatorBot itself* to move, not just the roomba. This changes the joint state to $s_t = (x_t, L_t)$ and gives the agent, at every step, a genuine choice between two categories of action: **sense**, or **move** (either the roomba toward the locator, or the locator toward the belief peak) — selected by comparing their expected information gain:

- **Expected entropy reduction from sensing:**
$$
\Delta H_{\text{sense}} = H(b_t^{-}) - \mathbb{E}_{z_t}\big[H(b_t \mid z_t)\big], \qquad H(b) = -\sum_x b(x)\log b(x)
$$
- **Expected certainty gain from repositioning the locator toward the belief peak:**
$$
\Delta C_{\text{move}} = C_t(L_t') - C_t(L_t)
$$
- **Action selection** — take whichever action maximizes expected utility net of its cost:
$$
a_t^{*} = \arg\max_{a_t} \big(\mathbb{E}[U_t(a_t)] - \text{Cost}(a_t)\big), \quad
U_t = -\big(w_m^{(R)} M_t^{(R)} + w_s S_t + w_m^{(L)} M_t^{(L)} + w_u(1 - C_t)\big)
$$

So: **if $\Delta H_{\text{sense}} > \Delta C_{\text{move}}$**, run the detector — sensing currently carries more information than repositioning would. **Otherwise**, move the LocatorBot toward the region of highest belief mass.

Allowing the locator to move helps precisely because *proximity is what makes the detector informative* — the exponential decay model means a beep/no-beep reading from close range is far less ambiguous than the same reading from far away. By repositioning itself, the LocatorBot also breaks spatial symmetry in the belief (cells equidistant from a stationary locator are indistinguishable to the sensor; a moving locator resolves that ambiguity). Movement isn't automatic, though — the utility framework only moves the locator when the expected certainty gain outweighs the extra movement cost, so the strategy behaves adaptively rather than always relocating.

---

## 4. Design Rationale — Q&A Writeup

### Belief update formulas for motion and sensing

The full motion-update law is Eq. 1–2 above: the predicted belief at a candidate cell $b$ is the transition-weighted sum over all prior candidate cells $a$, with an 80%/20% success/bounce transition model that also naturally covers the case where the intended move lands the roomba directly on the LocatorBot (the run simply terminates at that point). The sensor-update law is Eq. 3–5: the posterior belief is the prior belief reweighted by the likelihood of the observed beep/no-beep outcome under the exponential-decay detector model, then renormalized. Every strategy in this repo is a different policy for *when* to apply which of these two updates and what action to take between them.


### Design choices behind the Developed Strategy

The Developed Strategy replaces Baseline 2's "always chase the MAP cell" rule with an explicit utility function (Eq. 6) that prices moves, senses, and residual uncertainty against each other, and adds an exploration bias that issues disambiguating random moves when belief certainty is still low rather than prematurely converging on a shaky peak. This is theoretically grounded in utility theory: because the utility function is complete, transitive, continuous, and additively decomposable across its cost terms, comparing actions by expected utility is a principled decision rule rather than a heuristic guess — and it's this framework that lets the strategy recognize *when* sensor information has become reliable enough to act on decisively.


### Does letting the LocatorBot move help, and why?

Yes — allowing the LocatorBot to move generally reduces total actions, especially when $\alpha$ is small (a weak detector) or the ship is large, because moving closer to the roomba directly increases the sensor's effective signal-to-noise ratio (the exponential-decay model makes close-range beeps far more discriminating than far-range ones) and breaks the spatial symmetry that otherwise leaves many equidistant cells indistinguishable to a stationary detector. Movement is not unconditional, though: the entropy-vs-certainty comparison in Eq. 12–14 means the LocatorBot only relocates when the expected uncertainty reduction from moving outweighs its cost, so it behaves adaptively rather than wastefully repositioning every step.


*(Full Q4 performance write-up, including the α-sweep graphs for all four strategies, lives in [`/results`](./results).)*

---

## 5. Results

Experiments were run on a 15×15 ship (seed 85), with the LocatorBot fixed at `(2,2)` and the roomba starting at `(10,10)`, sweeping detector sensitivity $\alpha \in \{0.05, 0.1, 0.2, 0.3, 0.4, 0.5\}$. Full plots — total actions vs. $\alpha$ for all strategies, and the separate moves/senses breakdown — are in [`/results`](./results).

| α (detector sensitivity) | Baseline 2 (total actions) | Developed Strategy (total actions) |
|:---:|:---:|:---:|
| 0.05 – 0.4 | ~400 (capped — detector too noisy to converge) | ~400 (capped — detector too noisy to converge) |
| **0.5** | **400** | **36** (normalized utility ≈ **0.514**) |

**Reading the table:** at low $\alpha$, neither strategy can do much better than the action cap — with a sensor this noisy, there simply isn't enough information per reading for *any* policy to converge quickly, so the two strategies perform similarly. The gap opens up sharply once the detector becomes moderately informative ($\alpha = 0.5$): Baseline 2's greedy MAP-chasing still fails to converge within the cap, while the Developed Strategy's decision-theoretic action selection collapses the belief in 36 total actions — over 10× fewer. This is the core empirical argument for reasoning about *expected utility of information* rather than reflexively chasing the current best guess.



