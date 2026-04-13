import kaggle_benchmarks as kbench
import random, re


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

LL_N      = 4
TEST_REPS = 8


@kbench.task(
    name="List Linking - Non-Probe Cross-List Inference",
    description=(
        "Two separate 4-item ranked lists with a single linking probe connecting "
        "them. Tests all 16 cross-list pairs (8 reps). Key metric: non-probe "
        "accuracy on pairs not involving the two directly linked items."
    ),
)
def list_linking_non_probe(llm) -> None:
    all_labels  = [f"Item-{chr(65+i)}" for i in range(2 * LL_N)]
    random.shuffle(all_labels)
    list1       = all_labels[:LL_N]
    list2       = all_labels[LL_N:]
    D, E        = list1[-1], list2[0]
    probe_items = {D, E}

    list1_rules = "\n".join(
        f"  {list1[i]} ranks higher than {list1[i+1]}" for i in range(LL_N - 1)
    )
    list2_rules = "\n".join(
        f"  {list2[i]} ranks higher than {list2[i+1]}" for i in range(LL_N - 1)
    )

    # Single preamble — same pattern as working SDE task
    llm.prompt(
        f"LIST LINKING TASK\n"
        f"List 1 rules:\n{list1_rules}\n\n"
        f"List 2 rules:\n{list2_rules}\n\n"
        f"Linking rule: {D} ranks higher than {E}\n\n"
        f"All rules are transitive. The two lists are now fully connected.\n"
        f"Reply with ONLY the item name that ranks higher. No other text."
    )

    cross_pairs     = [(a, b) for a in list1 for b in list2]
    non_probe_pairs = [(a, b) for a, b in cross_pairs
                       if a not in probe_items and b not in probe_items]

    all_ch, all_tg = [], []
    np_ch,  np_tg  = [], []

    for _ in range(TEST_REPS):
        for a, b in cross_pairs:
            order  = [a, b]; random.shuffle(order)
            resp   = llm.prompt(f"Which ranks higher: {order[0]} or {order[1]}?")
            choice = _parse(resp, order[0], order[1])
            all_ch.append(choice); all_tg.append(a)

        for a, b in non_probe_pairs:
            order  = [a, b]; random.shuffle(order)
            resp   = llm.prompt(f"Which ranks higher: {order[0]} or {order[1]}?")
            choice = _parse(resp, order[0], order[1])
            np_ch.append(choice); np_tg.append(a)

    overall_acc  = _acc(all_ch,  all_tg)
    nonprobe_acc = _acc(np_ch,   np_tg)

    kbench.assertions.assert_true(
        nonprobe_acc >= 0.65,
        expectation=f"Non-probe accuracy should exceed chance (>=0.65). Got {nonprobe_acc:.3f}."
    )

list_linking_non_probe.run(kbench.llm)
