import kaggle_benchmarks as kbench
import random, re
import numpy as np


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

def _parse_learnability(response):
    text = response.strip().lower()
    if re.search(r'\bdifferent\b', text): return "different"
    if re.search(r'\bsame\b', text):      return "same"
    return "not sure"

def _slope(x, y):
    if len(x) < 2: return 0.0
    x, y = np.array(x, float), np.array(y, float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0

N_TRAIN_BLK = 3
N_TEST_BLK  = 3


@kbench.task(
    name="Learnability (PN) - Ordering Preference Discrimination",
    description=(
        "Two interleaved 5-item stimulus sets. Learnable (L) items have a "
        "consistent hidden ranking. Unlearnable (U) items use PN schedule. "
        "Key metric: ||OP_L|-|OP_U|| > 0.1 and OP_L < -0.15. "
        "Also asks model to self-report whether the two groups felt same/different."
    ),
)
def learnability_pn_op_discrimination(llm) -> None:
    base   = [f"Stim-{chr(65+i)}" for i in range(10)]
    random.shuffle(base)
    pool_L = base[:5]
    pool_U = base[5:]

    adj_L = [(pool_L[i], pool_L[i+1]) for i in range(4)]
    adj_U = [(pool_U[i], pool_U[i+1]) for i in range(4)]
    all_L = [(pool_L[i], pool_L[j]) for i in range(5) for j in range(i+1, 5)]
    all_U = [(pool_U[i], pool_U[j]) for i in range(5) for j in range(i+1, 5)]

    trials = []
    for _ in range(N_TRAIN_BLK):
        blk = []
        for a, b in adj_L:
            blk += [{"is_L": True,  "test": False, "a": a, "b": b},
                    {"is_L": True,  "test": False, "a": b, "b": a}]
        for a, b in adj_U:
            blk += [{"is_L": False, "test": False, "a": a, "b": b},
                    {"is_L": False, "test": False, "a": b, "b": a}]
        random.shuffle(blk)
        trials.extend(blk)
    for _ in range(N_TEST_BLK):
        blk = []
        for a, b in all_L:
            blk += [{"is_L": True,  "test": True, "a": a, "b": b},
                    {"is_L": True,  "test": True, "a": b, "b": a}]
        for a, b in all_U:
            blk += [{"is_L": False, "test": True, "a": a, "b": b},
                    {"is_L": False, "test": True, "a": b, "b": a}]
        random.shuffle(blk)
        trials.extend(blk)

    # Single preamble — tiny per-trial prompts keep cost low
    llm.prompt(
        f"LEARNABILITY TASK\n"
        f"Group L: {', '.join(pool_L)}\n"
        f"Group U: {', '.join(pool_U)}\n"
        f"Group L has a consistent hidden ranking. "
        f"Each trial: I show the previous outcome then two stimuli. "
        f"Reply with ONLY the stimulus name you choose."
    )

    L_outcomes   = []
    appeared_L   = {item: 0 for item in pool_L}
    chosen_L     = {item: 0 for item in pool_L}
    appeared_U   = {item: 0 for item in pool_U}
    chosen_U     = {item: 0 for item in pool_U}
    history_lines = []
    last_outcome  = None

    for t_num, trial in enumerate(trials):
        a, b    = trial["a"], trial["b"]
        is_L    = trial["is_L"]
        is_test = trial["test"]

        prefix = f"[{last_outcome}] " if last_outcome is not None else ""
        resp   = llm.prompt(f"{prefix}{a} vs {b}")
        choice = _parse(resp, a, b) or a

        if is_L:
            ra, rb   = pool_L.index(a), pool_L.index(b)
            target   = a if ra < rb else b
            rewarded = int(choice == target)
            L_outcomes.append(rewarded)
        else:
            prob     = float(np.mean(L_outcomes[-10:])) if len(L_outcomes) >= 10 else 0.5
            rewarded = int(random.random() < prob)

        last_outcome = "+reward" if rewarded else "no reward"
        history_lines.append(f"T{t_num+1}: {a} vs {b} -> {choice} ({last_outcome})")

        if is_test:
            if is_L:
                appeared_L[a] += 1; appeared_L[b] += 1
                if choice in chosen_L: chosen_L[choice] += 1
            else:
                appeared_U[a] += 1; appeared_U[b] += 1
                if choice in chosen_U: chosen_U[choice] += 1

    # ── OP slopes ──────────────────────────────────────────────────────────────
    cf_L = [chosen_L[item] / appeared_L[item] if appeared_L[item] > 0 else 0.5
            for item in pool_L]
    cf_U = [chosen_U[item] / appeared_U[item] if appeared_U[item] > 0 else 0.5
            for item in pool_U]
    op_L        = _slope(list(range(1, 6)), cf_L)
    op_U        = _slope(list(range(1, 6)), cf_U)
    op_disc     = op_L - op_U
    abs_op_disc = abs(abs(op_L) - abs(op_U))  # ||OP_L| - |OP_U||

    # ── Self-report: single cheap prompt with compact history ──────────────────
    compact_history = "\n".join(history_lines)
    self_report_resp = llm.prompt(
        f"Task complete. History:\n{compact_history}\n\n"
        f"Did Group L ({', '.join(pool_L)}) and Group U ({', '.join(pool_U)}) "
        f"feel SAME, DIFFERENT, or NOT SURE in terms of predictability?\n"
        f"Reply with one: same, different, not sure."
    )
    self_report = _parse_learnability(self_report_resp)

    summary = (
        f"OP_L={op_L:.4f}, OP_U={op_U:.4f}, "
        f"OP_discrimination={op_disc:.4f}, ||OP_L|-|OP_U||={abs_op_disc:.4f}, "
        f"self_report='{self_report}'"
    )

    kbench.assertions.assert_true(
        op_L < -0.15,
        expectation=(
            f"OP_L should be < -0.15 (clear ordering preference for L). "
            f"Got OP_L={op_L:.4f}. {summary}"
        )
    )
    kbench.assertions.assert_true(
        abs_op_disc > 0.1,
        expectation=(
            f"||OP_L|-|OP_U|| should be > 0.1 (clear separation between L and U). "
            f"Got {abs_op_disc:.4f}. {summary}"
        )
    )
    kbench.assertions.assert_true(
        self_report == "different",
        expectation=(
            f"Model should report groups as 'different'. "
            f"Got: '{self_report}'. {summary}"
        )
    )

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=summary,
        judge_llm=kbench.judge_llm,
        criteria=[
            "OP_L should be clearly negative (< -0.15), showing the model "
            "clearly preferred higher-ranked learnable stimuli.",
            "||OP_L| - |OP_U|| should exceed 0.1, showing meaningful behavioural "
            "separation between how the model treated Group L vs Group U.",
            "The model's self-report should be 'different', indicating it "
            "detected that one group was more structured than the other.",
        ]
    )
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge: '{result.criterion}': {result.reason}"
        )

learnability_pn_op_discrimination.run(kbench.llm)
