# Cognitive Benchmark Task Suite

Three tasks measuring structured reasoning and learnability in AI models, adapted from primate and computational neuroscience paradigms.

---

## Task 1: Transitive Inference (TI)

Transitive inference (TI) probes whether a model can infer the relative rank of two items it has never directly compared, by chaining through items it has compared. The classic paradigm trains subjects on adjacent pairs from a hidden linear hierarchy — A>B, B>C, C>D, D>E, E>F — and tests on non-adjacent pairs such as B>D, which requires implicit knowledge of the full ordering.

**Design**

- Hierarchy: 6 items, labels randomly shuffled per session so rank is not inferable from name
- Training: Adjacent pairs only (A–B, B–C … E–F), 15 reps each; outcome shown in prompt history
- Test: All non-adjacent pairs (distances 2–5), 8 reps each; full training history in context
- Response format: Model outputs the name of the higher-ranked item only

**Metrics**

- Symbolic distance effect (SDE slope): OLS slope of accuracy ~ distance (2..5). Positive = farther pairs are easier.
- Terminal item effect (TIE): terminal accuracy − interior accuracy. Endpoint pairs vs. both-interior pairs. Positive = classic TIE.

The symbolic distance effect (SDE) arises because items far apart in the hierarchy provide stronger inferential evidence. The terminal item effect (TIE) reflects that rank-1 always wins and rank-N always loses, making those pairs trivially easier. A model showing SDE but not TIE is more likely relying on genuine transitive reasoning; one showing TIE but not SDE may be anchoring on endpoints alone.

---

## Task 2: List Linking (LL)

List linking tests whether a model can integrate two separately-learned orderings into a unified transitive chain when given a single connecting fact. The paradigm originates in comparative cognition research asking whether primates can infer cross-list ranks after learning each list in isolation, using one explicit probe trial as a bridge.

**Design**

- List 1: A > B > C > D — trained on adjacent pairs (A–B, B–C, C–D), 15 reps each
- List 2: E > F > G > H — trained on adjacent pairs (E–F, F–G, G–H), 15 reps each
- Linking probe: D vs E → D wins, shown 5 times after both lists are trained
- Combined order: A > B > C > D > E > F > G > H
- Test: All 16 cross-list pairs (List-1 item vs List-2 item). Correct = always the List-1 item.

**Key metric: non-probe accuracy**

The critical measure is accuracy on cross-list pairs that do not involve D or E — the two items directly linked. Correctly ranking A above F, for instance, requires chaining A>C>D>E>F across three separate training and probe steps. Above-chance performance on these pairs is the definitive list-linking signature: the model formed a unified ordered representation rather than memorising the probe pair.

- Overall cross-list accuracy: all 16 pairs. List-1 item should always win. Includes pairs involving D and E.
- Non-probe accuracy: 9 pairs excluding D and E. The true list-linking test. Requires multi-step inference through the probe link.

---

## Task 3: Learnability Task (PN schedule)

The learnability task probes whether a model can detect that some stimulus sets have consistent learnable structure while others do not, and develop a preference ordering only for the former. It is adapted from monkey dACC electrophysiology experiments in which subjects learned to choose between stimuli under deterministic or probabilistic reward schedules, while neural activity was recorded to ask whether the brain encodes learnability as a separate signal from reward value.

**Stimulus sets**

Ten stimuli are divided into two groups of five with labels randomly shuffled per session. Group L (learnable) has a hidden consistent ranking: L1>L2>L3>L4>L5. Choosing the higher-ranked item is always rewarded. Group U (unlearnable) has no consistent ranking; reward under the PN schedule is determined purely by recent L performance, not by the model's choice.

**PN reward schedule**

- Learnable (L): Reward = +1 if higher-ranked item chosen, −1 otherwise. Consistent signal every trial.
- Unlearnable (U): P(reward) = rolling mean of the last 10 L outcomes. Reward rate for U is yoked to L performance, equating average feedback across both groups while removing any learnable structure from U.

The yoking design is the key control: it ensures that any difference in the model's behaviour toward L vs U items cannot be attributed to differences in average reward rate. Only structural learnability differs.

**Trial structure**

- Training blocks: 9 blocks of adjacent pairs only. L and U trials interleaved, shuffled within each block. Both orderings of each pair presented.
- Test blocks: 10 blocks of all pairs including non-adjacent. Tests whether ordering preference generalises beyond trained pairs.
- Context window: Full trial-by-trial history shown in prompt (up to 20 recent trials): stimulus names, choices, reward outcomes.

**Key metric: ordering preference (OP) slope**

For each item in a pool (L or U), compute its choose frequency across test trials only:

    choose_freq[item] = (times item chosen) / (times item appeared)

Fit a linear regression of choose frequency on true rank (rank 1 = best, rank 5 = worst):

    choose_freq ~ β₀ + β₁ × rank

The slope β₁ is the OP score. A negative slope means the model chose higher-ranked items more often — a correct ordering preference. A slope near zero means no consistent preference, as expected for unlearnable items. Three sessions with fresh random labels are run and averaged to reduce variance.

- OP_L: slope for learnable items. Should be clearly negative. Reflects ordering preference for learnable stimuli.
- OP_U: slope for unlearnable items. Should be near zero. Non-zero values suggest the model was misled by U feedback structure.
- OP_discrimination = OP_L − OP_U: the key learnability signature. More negative = model correctly distinguishes L from U.
