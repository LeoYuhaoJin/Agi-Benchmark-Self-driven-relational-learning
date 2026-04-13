import kaggle_benchmarks as kbench
import random, re
import numpy as np


def _shuffle_labels(n):
    pool = [f"Item-{chr(65+i)}" for i in range(26)]
    labels = pool[:n]; random.shuffle(labels); return labels

def _parse(response, a, b):
    text = response.strip().lower()
    for token, opt in [(a.lower(), a), (b.lower(), b)]:
        if re.search(r'\b' + re.escape(token) + r'\b', text):
            return opt
    pos_a, pos_b = text.find(a.lower()), text.find(b.lower())
    if pos_a >= 0 and pos_b >= 0: return a if pos_a < pos_b else b
    if pos_a >= 0: return a
    if pos_b >= 0: return b
    return None

def _acc(choices, targets):
    correct = sum(c == t for c, t in zip(choices, targets) if c is not None)
    valid   = sum(c is not None for c in choices)
    return correct / valid if valid else 0.0

def _slope(x, y):
    if len(x) < 2: return 0.0
    x, y = np.array(x, float), np.array(y, float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0

N_ITEMS    = 5
TRAIN_REPS = 15
TEST_REPS  = 8


@kbench.task(
    name="Transitive Inference - Symbolic Distance Effect",
    description=(
        "5-item linear hierarchy. All adjacent ranking rules given in one prompt. "
        "Tested on all non-adjacent pairs (15 train reps shown, 8 test reps). "
        "Primary metric: overall accuracy > chance. SDE slope reported as diagnostic."
    ),
)
def ti_symbolic_distance_effect(llm) -> None:
    labels = _shuffle_labels(N_ITEMS)

    adj_lines = "\n".join(
        f"  {labels[i]} ranks higher than {labels[i+1]}"
        for i in range(N_ITEMS - 1)
    )
    llm.prompt(
        f"TRANSITIVE INFERENCE TASK\n"
        f"Complete ranking rules (higher = better):\n"
        f"{adj_lines}\n\n"
        f"These rules are transitive. Reply with ONLY the item name that ranks higher. No other text."
    )

    test_pairs = [(i, j, j-i) for i in range(N_ITEMS) for j in range(i+2, N_ITEMS)]
    dist_ch: dict[int, list] = {}
    dist_tg: dict[int, list] = {}

    for _ in range(TEST_REPS):
        for hi, lo, dist in test_pairs:
            a, b   = labels[hi], labels[lo]
            order  = [a, b]; random.shuffle(order)
            resp   = llm.prompt(f"Which ranks higher: {order[0]} or {order[1]}?")
            choice = _parse(resp, order[0], order[1])
            dist_ch.setdefault(dist, []).append(choice)
            dist_tg.setdefault(dist, []).append(a)

    dists   = sorted(dist_ch.keys())
    accs    = [_acc(dist_ch[d], dist_tg[d]) for d in dists]
    slope   = _slope(dists, accs)
    overall = _acc(
        [c for cs in dist_ch.values() for c in cs],
        [t for ts in dist_tg.values() for t in ts]
    )
    dist_summary = ", ".join(f"dist={d}: {a:.3f}" for d, a in zip(dists, accs))

    kbench.assertions.assert_true(
        overall >= 0.65,
        expectation=f"TI accuracy should exceed chance (>=0.65). Got {overall:.3f}. {dist_summary}"
    )

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=(
            f"Transitive inference: overall={overall:.3f}, {dist_summary}, "
            f"SDE slope={slope:.4f} (0.0 at ceiling is valid)"
        ),
        judge_llm=kbench.judge_llm,
        criteria=[
            "Overall accuracy should exceed chance (>0.65), showing the model "
            "can infer rankings between items never directly compared.",
        ]
    )
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge: '{result.criterion}': {result.reason}"
        )

ti_symbolic_distance_effect.run(kbench.llm)
