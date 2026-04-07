"""Deterministic graders for OpenEnv email triage tasks."""

import re

from models import RewardResult, TriageAction

ROUTE_ALIAS_MAP = {
    "billing": ["billing", "finance", "payments", "accounts"],
    "safety": ["safety", "compliance", "risk"],
    "engineering": ["engineering", "eng", "sre", "platform", "on-call"],
    "support": ["support", "helpdesk", "customer support"],
    "general": ["general", "inbox", "operations"],
}

SCORE_EPSILON = 1e-6


def _strict_binary_score(is_positive_case: bool) -> float:
    """Return strict in-range score for binary outcomes."""
    return 1.0 - SCORE_EPSILON if is_positive_case else SCORE_EPSILON


def _strict_ratio_score(raw_value: float) -> float:
    """Return strict in-range score for ratio-like metrics."""
    return _clip_score(raw_value)


def _clip_score(score_value: float) -> float:
    """Clip a score to the strict range (0.0, 1.0).

    Args:
        score_value: Raw score.

    Returns:
        Clipped score.
    """
    clipped = max(0.0, min(1.0, score_value))
    if clipped <= 0.0:
        return SCORE_EPSILON
    if clipped >= 1.0:
        return 1.0 - SCORE_EPSILON
    return clipped


def _normalized_text(text_value: str) -> str:
    """Return normalized lowercase text for deterministic comparisons.

    Args:
        text_value: Input text.

    Returns:
        Normalized text.
    """
    return text_value.strip().lower()


def _route_matches(action_route: str, expected_route: str) -> bool:
    """Check if action route contains the expected route token.

    Args:
        action_route: Route provided by agent.
        expected_route: Route expected by ground truth.

    Returns:
        True when expected route is present in the action route.
    """
    normalized_expected = _normalized_text(expected_route)
    if not normalized_expected:
        return False

    return normalized_expected in _canonical_route_tokens(action_route)


def _canonical_route_tokens(action_route: str) -> set[str]:
    """Map free-form route text to canonical route categories."""
    normalized_action = _normalized_text(action_route)
    if not normalized_action:
        return set()

    route_fragments = [
        fragment.strip()
        for fragment in re.split(r"[,;/|]+", normalized_action)
        if fragment.strip()
    ]

    canonical: set[str] = set()
    for fragment in route_fragments:
        for route_name, aliases in ROUTE_ALIAS_MAP.items():
            if any(alias in fragment for alias in aliases):
                canonical.add(route_name)
                break

    # Fallback for phrases without separators.
    if not canonical:
        for route_name, aliases in ROUTE_ALIAS_MAP.items():
            if any(alias in normalized_action for alias in aliases):
                canonical.add(route_name)

    return canonical


def _route_noise_penalty(action_route: str) -> float:
    """Penalize over-routing to many teams in one action."""
    route_count = len(_canonical_route_tokens(action_route))
    if route_count <= 2:
        return 0.0
    return min(0.24, 0.08 * (route_count - 2))


def _summary_keyword_score(summary_text: str, ground_truth: dict) -> float:
    """Score summary quality using deterministic keyword overlap.

    Args:
        summary_text: Summary text produced by the agent.
        ground_truth: Ground-truth dict that may include summary keywords.

    Returns:
        Score in [0.0, 1.0] based on matched summary keywords.
    """
    raw_keywords = ground_truth.get("summary_keywords", [])
    if not isinstance(raw_keywords, list):
        return _strict_binary_score(len(summary_text.strip()) >= 10)

    keywords = [
        _normalized_text(str(keyword))
        for keyword in raw_keywords
        if _normalized_text(str(keyword))
    ]
    if not keywords:
        return _strict_binary_score(len(summary_text.strip()) >= 10)

    normalized_summary = _normalized_text(summary_text)
    matches = 0
    for keyword in keywords:
        if keyword in normalized_summary:
            matches += 1

    base_score = matches / len(keywords)

    # Discourage keyword stuffing and overly verbose summaries.
    word_count = len(re.findall(r"[a-z0-9'-]+", normalized_summary))
    if word_count < 4:
        brevity_factor = 0.6
    elif word_count <= 40:
        brevity_factor = 1.0
    else:
        brevity_factor = max(0.45, 1.0 - (word_count - 40) * 0.02)

    list_like_penalty = 0.85 if normalized_summary.count(",") >= 6 and matches >= 3 else 1.0
    return _clip_score(base_score * brevity_factor * list_like_penalty)


def grade_easy(action: TriageAction, ground_truth: dict) -> RewardResult:
    """Grade easy task with deterministic partial credit.

    Args:
        action: Agent action for one email.
        ground_truth: Expected label and route.

    Returns:
        Deterministic reward result in [0.0, 1.0].
    """
    expected_label = _normalized_text(str(ground_truth.get("label", "")))
    expected_route = _normalized_text(str(ground_truth.get("route_to", "")))

    label_correct = _normalized_text(action.label) == expected_label
    route_correct = _route_matches(action.route_to, expected_route)
    summary_score = _summary_keyword_score(action.summary, ground_truth)
    noise_penalty = _route_noise_penalty(action.route_to)

    score_value = (0.6 if label_correct else 0.0) + (0.25 if route_correct else 0.0)
    score_value += 0.15 * summary_score
    score_value -= noise_penalty

    score_value = _clip_score(score_value)
    breakdown = {
        "label_match": _strict_binary_score(label_correct),
        "route_match": _strict_binary_score(route_correct),
        "summary_match": _strict_ratio_score(summary_score),
        "route_noise_penalty": _strict_ratio_score(noise_penalty),
    }
    feedback = "Easy-task grading completed with context summary scoring."
    return RewardResult(score=score_value, breakdown=breakdown, feedback=feedback)


def grade_medium_step(action: TriageAction, truth: dict) -> RewardResult:
    """Grade one medium-task step without cumulative history effects."""
    expected_label = _normalized_text(str(truth.get("label", "")))
    expected_route = _normalized_text(str(truth.get("route_to", "")))
    priority_weight = max(float(truth.get("priority_weight", 1.0)), 0.1)

    label_correct = _normalized_text(action.label) == expected_label
    route_correct = _route_matches(action.route_to, expected_route)
    summary_score = _summary_keyword_score(action.summary, truth)
    noise_penalty = _route_noise_penalty(action.route_to)

    per_email_score = (0.55 if label_correct else 0.0) + (0.3 if route_correct else 0.0)
    per_email_score += 0.15 * summary_score
    per_email_score -= noise_penalty
    per_email_score = _clip_score(per_email_score)

    weighted_step_score = _clip_score(per_email_score * min(priority_weight, 2.0))

    return RewardResult(
        score=weighted_step_score,
        breakdown={
            "label_match": _strict_binary_score(label_correct),
            "route_match": _strict_binary_score(route_correct),
            "summary_match": _strict_ratio_score(summary_score),
            "priority_weight": _strict_ratio_score(min(priority_weight / 2.0, 1.0)),
            "route_noise_penalty": _strict_ratio_score(noise_penalty),
        },
        feedback="Medium-task step grading completed.",
    )


def grade_medium(actions: list[TriageAction], ground_truths: list[dict]) -> RewardResult:
    """Grade medium task using weighted per-email partial scoring.

    Args:
        actions: Agent actions for the medium task email queue.
        ground_truths: Expected action details for each email.

    Returns:
        Deterministic reward result in [0.0, 1.0].
    """
    comparable_count = min(len(actions), len(ground_truths))
    if comparable_count == 0:
        return RewardResult(
            score=SCORE_EPSILON,
            breakdown={"emails_scored": SCORE_EPSILON, "weighted_average": SCORE_EPSILON},
            feedback="No actions available for grading.",
        )

    weighted_score_sum = 0.0
    weight_sum = 0.0
    label_hits = 0
    route_hits = 0
    summary_total = 0.0
    noise_penalty_total = 0.0

    for index in range(comparable_count):
        action = actions[index]
        truth = ground_truths[index]

        step_result = grade_medium_step(action, truth)
        priority_weight = max(float(truth.get("priority_weight", 1.0)), 0.1)
        weighted_score_sum += step_result.score
        weight_sum += min(priority_weight, 2.0)

        expected_label = _normalized_text(str(truth.get("label", "")))
        expected_route = _normalized_text(str(truth.get("route_to", "")))
        label_hits += 1 if _normalized_text(action.label) == expected_label else 0
        route_hits += 1 if _route_matches(action.route_to, expected_route) else 0
        summary_total += float(step_result.breakdown.get("summary_match", SCORE_EPSILON))
        noise_penalty_total += float(
            step_result.breakdown.get("route_noise_penalty", SCORE_EPSILON)
        )

    weighted_average = weighted_score_sum / weight_sum if weight_sum > 0.0 else 0.0
    score_value = _clip_score(weighted_average)

    breakdown = {
        "emails_scored": _strict_ratio_score(float(comparable_count) / (comparable_count + 1.0)),
        "label_accuracy": _strict_ratio_score(label_hits / comparable_count),
        "route_accuracy": _strict_ratio_score(route_hits / comparable_count),
        "summary_accuracy": _strict_ratio_score(summary_total / comparable_count),
        "avg_route_noise_penalty": _strict_ratio_score(noise_penalty_total / comparable_count),
        "weighted_average": score_value,
    }
    feedback = "Weighted medium-task grading completed."
    return RewardResult(score=score_value, breakdown=breakdown, feedback=feedback)


def grade_hard(action: TriageAction, ground_truth: dict) -> RewardResult:
    """Grade hard task using weighted policy-sensitive components.

    Args:
        action: Agent action for hard task case.
        ground_truth: Expected routing and urgency intent.

    Returns:
        Deterministic reward result in [0.0, 1.0].
    """
    expected_label = _normalized_text(str(ground_truth.get("label", "urgent")))
    primary_route = _normalized_text(str(ground_truth.get("route_to", "safety")))
    secondary_route = _normalized_text(str(ground_truth.get("cc_route", "billing")))
    spam_penalty = float(ground_truth.get("penalize_spam", 0.2))

    normalized_route = _normalized_text(action.route_to)
    has_primary_route = _route_matches(normalized_route, primary_route)
    has_secondary_route = _route_matches(normalized_route, secondary_route)
    urgent_label = _normalized_text(action.label) == expected_label
    summary_score = _summary_keyword_score(action.summary, ground_truth)
    noise_penalty = _route_noise_penalty(action.route_to)

    escalation_component = 0.35 if has_primary_route else 0.0
    routing_component = 0.25 if has_secondary_route else 0.0
    urgency_component = 0.25 if urgent_label else 0.0
    summary_component = 0.15 * summary_score

    raw_score = escalation_component + routing_component + urgency_component + summary_component
    raw_score -= noise_penalty
    if _normalized_text(action.label) == "spam":
        raw_score -= spam_penalty

    score_value = _clip_score(raw_score)
    breakdown = {
        "escalation_component": _strict_ratio_score(escalation_component),
        "routing_component": _strict_ratio_score(routing_component),
        "urgency_component": _strict_ratio_score(urgency_component),
        "summary_component": _strict_ratio_score(summary_component),
        "route_noise_penalty": _strict_ratio_score(noise_penalty),
        "spam_penalty": _strict_ratio_score(
            spam_penalty if _normalized_text(action.label) == "spam" else SCORE_EPSILON
        ),
    }
    feedback = "Hard-task weighted policy grading completed."
    return RewardResult(score=score_value, breakdown=breakdown, feedback=feedback)
