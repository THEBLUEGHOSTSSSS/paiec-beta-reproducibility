"""Fixed Beta(1, 1) posterior mean; untrained PAIEC comparison candidate.

Uses only the provided labels for the exact visible subject and benchmark.
No network, learned data, cross-call state, or acquisition hook.
"""


def predict(input: list[dict], labeled: list | None = None) -> float:
    subject, item = input
    if not labeled:
        return 0.5
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("Positive-label calls require benchmark_id")
    successes, count = 0, 0
    for (observed_subject, observed_item), outcome in labeled:
        if observed_subject != subject:
            continue
        observed_benchmark = observed_item.get("benchmark_id")
        if not isinstance(observed_benchmark, str) or not observed_benchmark:
            raise ValueError("Acquired evidence requires benchmark_id")
        if observed_benchmark != benchmark:
            continue
        if outcome not in (0, 1):
            raise ValueError("Matching outcomes must be binary")
        successes += int(outcome)
        count += 1
    return (successes + 1.0) / (count + 2.0)
