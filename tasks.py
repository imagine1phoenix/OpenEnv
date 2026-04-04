"""Task definitions and scenario pools for the OpenEnv email triage environment."""

import json
import os
import random
from datetime import datetime, timedelta, timezone

from typing import cast

TASK_LIBRARY: dict[str, dict[str, object]] = {
    "task_easy": {
        "description": "Classify and route one unambiguous operational email.",
        "scenario_pool": [
            {
                "scenario_id": "easy_invoice_confirmation",
                "emails": [
                    {
                        "email_id": "easy-001",
                        "subject": "Quarterly invoice available",
                        "body": (
                            "Hello Team, your Q1 invoice is now ready in the billing portal. "
                            "Please confirm the purchase order number by Friday."
                        ),
                        "sender": "accounts@vendor-example.com",
                        "timestamp": "2026-03-25T09:15:00Z",
                        "thread_history": [
                            "Last month: requested invoice schedule for Q1 and Q2."
                        ],
                    }
                ],
                "ground_truth": [
                    {
                        "label": "normal",
                        "route_to": "billing",
                        "priority_weight": 1.0,
                        "summary_keywords": ["invoice", "purchase order", "billing portal"],
                    }
                ],
            },
            {
                "scenario_id": "easy_password_lockout",
                "emails": [
                    {
                        "email_id": "easy-002",
                        "subject": "Locked out of admin dashboard",
                        "body": (
                            "I cannot access the admin dashboard after MFA reset, and our "
                            "client demo starts in 20 minutes. Please help immediately."
                        ),
                        "sender": "sales-lead@acme-enterprise.com",
                        "timestamp": "2026-03-28T11:40:00Z",
                        "thread_history": ["MFA reset was completed this morning."],
                    }
                ],
                "ground_truth": [
                    {
                        "label": "urgent",
                        "route_to": "support",
                        "priority_weight": 1.2,
                        "summary_keywords": ["locked out", "mfa", "demo"],
                    }
                ],
            },
            {
                "scenario_id": "easy_newsletter_archive",
                "emails": [
                    {
                        "email_id": "easy-003",
                        "subject": "Monthly partner newsletter",
                        "body": (
                            "Sharing this month's partner newsletter with product updates. "
                            "No action needed unless you want to read the highlights."
                        ),
                        "sender": "updates@partner-network.io",
                        "timestamp": "2026-03-30T08:10:00Z",
                        "thread_history": [],
                    }
                ],
                "ground_truth": [
                    {
                        "label": "archive",
                        "route_to": "general",
                        "priority_weight": 0.8,
                        "summary_keywords": ["newsletter", "no action", "updates"],
                    }
                ],
            },
        ],
        "private_eval_pool": [],
    },
    "task_medium": {
        "description": "Triage five mixed-priority emails with ambiguous contextual signals.",
        "scenario_pool": [
            {
                "scenario_id": "medium_ops_mix_a",
                "emails": [
                    {
                        "email_id": "med-001",
                        "subject": "URGENT: Your account will be disabled in 30 minutes",
                        "body": (
                            "Click this external short link to keep your mailbox active. "
                            "If you do not click now, your account will be deleted."
                        ),
                        "sender": "it-admin@secure-mail-help.net",
                        "timestamp": "2026-03-26T07:08:00Z",
                        "thread_history": [],
                    },
                    {
                        "email_id": "med-002",
                        "subject": "Can someone review production error spikes?",
                        "body": (
                            "We are seeing a 28% spike in checkout failures after the 06:10 UTC "
                            "deploy. Please triage and assign on-call ownership immediately."
                        ),
                        "sender": "ops-manager@acme-enterprise.com",
                        "timestamp": "2026-03-26T06:21:00Z",
                        "thread_history": ["Pager alert opened at 06:18 UTC."],
                    },
                    {
                        "email_id": "med-003",
                        "subject": "RE: promo campaign winner list",
                        "body": (
                            "Subject line looks like a campaign thread, but this message confirms "
                            "a customer reported duplicate card charges. Please review and respond."
                        ),
                        "sender": "care-escalations@acme-enterprise.com",
                        "timestamp": "2026-03-26T11:42:00Z",
                        "thread_history": [
                            "Marketing team forwarded customer complaint for billing review."
                        ],
                    },
                    {
                        "email_id": "med-004",
                        "subject": "Safety escalation: charger overheating case #4812",
                        "body": (
                            "Customer reports visible smoke from charging dock during normal use. "
                            "No injuries reported, but immediate safety review requested."
                        ),
                        "sender": "support-lead@acme-enterprise.com",
                        "timestamp": "2026-03-26T10:03:00Z",
                        "thread_history": ["Ticket severity raised from P2 to P1."],
                    },
                    {
                        "email_id": "med-005",
                        "subject": "FYI: April all-hands agenda",
                        "body": (
                            "Sharing the all-hands agenda draft. No action required unless you "
                            "want to propose additional topics by Monday."
                        ),
                        "sender": "people-ops@acme-enterprise.com",
                        "timestamp": "2026-03-26T14:25:00Z",
                        "thread_history": [],
                    },
                ],
                "ground_truth": [
                    {
                        "label": "spam",
                        "route_to": "general",
                        "priority_weight": 1.0,
                        "summary_keywords": ["external link", "disable", "phishing"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "engineering",
                        "priority_weight": 1.5,
                        "summary_keywords": ["checkout", "failures", "on-call"],
                    },
                    {
                        "label": "normal",
                        "route_to": "billing",
                        "priority_weight": 1.2,
                        "summary_keywords": ["duplicate", "charges", "billing"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "priority_weight": 1.6,
                        "summary_keywords": ["smoke", "overheating", "safety"],
                    },
                    {
                        "label": "archive",
                        "route_to": "general",
                        "priority_weight": 0.8,
                        "summary_keywords": ["all-hands", "agenda", "no action required"],
                    },
                ],
            },
            {
                "scenario_id": "medium_ops_mix_b",
                "emails": [
                    {
                        "email_id": "med-b-001",
                        "subject": "Action required: verify payroll account immediately",
                        "body": (
                            "Your payroll account appears locked. Verify your credentials on this "
                            "new portal link to avoid delayed salary processing."
                        ),
                        "sender": "payroll-security@alerts-payroll.net",
                        "timestamp": "2026-04-01T07:00:00Z",
                        "thread_history": [],
                    },
                    {
                        "email_id": "med-b-002",
                        "subject": "Incident: checkout API timeout in eu-west",
                        "body": (
                            "Payments API timeout crossed SLO in eu-west for 11 minutes. "
                            "Revenue impact probable. On-call escalation required."
                        ),
                        "sender": "sre@acme-enterprise.com",
                        "timestamp": "2026-04-01T06:40:00Z",
                        "thread_history": ["Auto-remediation attempt failed."],
                    },
                    {
                        "email_id": "med-b-003",
                        "subject": "Question about duplicate invoice #4421",
                        "body": (
                            "Customer says invoice #4421 appears twice in the portal and asks "
                            "which one should be paid."
                        ),
                        "sender": "support@acme-enterprise.com",
                        "timestamp": "2026-04-01T09:22:00Z",
                        "thread_history": ["Ticket tagged as finance inquiry."],
                    },
                    {
                        "email_id": "med-b-004",
                        "subject": "Potential safety issue in warehouse charging bay",
                        "body": (
                            "Night shift observed sparks from charging rack slot C after routine use. "
                            "Operations paused affected rack."
                        ),
                        "sender": "warehouse-lead@acme-enterprise.com",
                        "timestamp": "2026-04-01T04:50:00Z",
                        "thread_history": ["Photos attached in internal ticket."],
                    },
                    {
                        "email_id": "med-b-005",
                        "subject": "Reminder: volunteer signup closes Friday",
                        "body": "Friendly reminder to sign up for community day volunteer slots.",
                        "sender": "people-ops@acme-enterprise.com",
                        "timestamp": "2026-04-01T10:12:00Z",
                        "thread_history": [],
                    },
                ],
                "ground_truth": [
                    {
                        "label": "spam",
                        "route_to": "general",
                        "priority_weight": 1.0,
                        "summary_keywords": ["verify", "portal link", "payroll"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "engineering",
                        "priority_weight": 1.5,
                        "summary_keywords": ["timeout", "slo", "on-call"],
                    },
                    {
                        "label": "normal",
                        "route_to": "billing",
                        "priority_weight": 1.2,
                        "summary_keywords": ["duplicate invoice", "paid", "finance"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "priority_weight": 1.6,
                        "summary_keywords": ["sparks", "charging", "paused"],
                    },
                    {
                        "label": "archive",
                        "route_to": "general",
                        "priority_weight": 0.8,
                        "summary_keywords": ["reminder", "volunteer", "signup"],
                    },
                ],
            },
        ],
        "private_eval_pool": [],
    },
    "task_hard": {
        "description": "Handle ambiguous complaints that mix safety, legal, and billing risk.",
        "scenario_pool": [
            {
                "scenario_id": "hard_cross_function_a",
                "emails": [
                    {
                        "email_id": "hard-001",
                        "subject": "Formal complaint: unsafe device behavior and disputed charges",
                        "body": (
                            "I was charged twice for the replacement kit, and during testing the "
                            "unit became hot enough to scorch the desk surface. I need billing "
                            "correction and urgent safety follow-up today."
                        ),
                        "sender": "legal-customer@enterprise-client.com",
                        "timestamp": "2026-03-26T08:33:00Z",
                        "thread_history": [
                            "Support asked customer to share photos; customer replied with incident details."
                        ],
                    },
                    {
                        "email_id": "hard-002",
                        "subject": "Escalation follow-up: compliance and refund timeline",
                        "body": (
                            "Following up on the same incident, compliance team requests confirmation "
                            "of safety escalation and billing refund timeline before we close the case."
                        ),
                        "sender": "procurement@enterprise-client.com",
                        "timestamp": "2026-03-26T09:07:00Z",
                        "thread_history": ["Legal requested cross-team response within 4 business hours."],
                    },
                ],
                "ground_truth": [
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "cc_route": "billing",
                        "penalize_spam": 0.2,
                        "summary_keywords": ["unsafe", "overheating", "double charge", "refund"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "cc_route": "billing",
                        "penalize_spam": 0.2,
                        "summary_keywords": ["compliance", "safety escalation", "refund timeline"],
                    },
                ],
            },
            {
                "scenario_id": "hard_cross_function_b",
                "emails": [
                    {
                        "email_id": "hard-b-001",
                        "subject": "Executive escalation: battery smoke + invoice mismatch",
                        "body": (
                            "A strategic customer reported smoke from a battery dock and also found an "
                            "invoice mismatch on the emergency replacement shipment."
                        ),
                        "sender": "exec-office@acme-enterprise.com",
                        "timestamp": "2026-04-01T07:40:00Z",
                        "thread_history": ["Account owner requested immediate cross-functional response."],
                    },
                    {
                        "email_id": "hard-b-002",
                        "subject": "Legal asks for mitigation proof and refund confirmation",
                        "body": (
                            "Legal team needs written mitigation steps for the safety issue and proof "
                            "that refund correction has been processed."
                        ),
                        "sender": "legal@enterprise-client.com",
                        "timestamp": "2026-04-01T08:05:00Z",
                        "thread_history": ["Deadline: before board review this afternoon."],
                    },
                ],
                "ground_truth": [
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "cc_route": "billing",
                        "penalize_spam": 0.2,
                        "summary_keywords": ["smoke", "invoice mismatch", "strategic customer"],
                    },
                    {
                        "label": "urgent",
                        "route_to": "safety",
                        "cc_route": "billing",
                        "penalize_spam": 0.2,
                        "summary_keywords": ["mitigation", "refund confirmation", "legal"],
                    },
                ],
            },
        ],
        "private_eval_pool": [],
    },
    "task_production": {
        "description": (
            "Simulate a production inbox with mixed business-critical, safety, billing, "
            "support, and spam threads."
        ),
        "scenario_pool": [],
        "private_eval_pool": [],
    },
}

PRODUCTION_SCENARIO_COUNT = 1000

PRODUCTION_TEMPLATE_LIBRARY: dict[str, dict[str, object]] = {
    "phishing_link": {
        "label": "spam",
        "route_to": "general",
        "priority_weight": 1.0,
        "summary_keywords": ["phishing", "external link", "credential"],
        "subject_options": [
            "Immediate verification required for shared inbox",
            "Mailbox suspension warning - action needed",
        ],
        "body_options": [
            "Security team asks you to verify credentials using this external short link.",
            "Your mailbox will be disabled unless you confirm account details on new portal.",
        ],
        "sender_options": ["security-alert@mail-checkup.net", "admin-support@auth-updates.co"],
    },
    "incident_checkout": {
        "label": "urgent",
        "route_to": "engineering",
        "priority_weight": 1.8,
        "summary_keywords": ["checkout", "incident", "on-call"],
        "subject_options": [
            "SEV-1: checkout failures rising",
            "Revenue incident: payment flow degraded",
        ],
        "body_options": [
            "Checkout success rate dropped after deployment and on-call escalation is pending.",
            "Payment API latency breach is impacting order completion in production.",
        ],
        "sender_options": ["incident-bot@acme-enterprise.com", "sre-lead@acme-enterprise.com"],
    },
    "billing_refund": {
        "label": "normal",
        "route_to": "billing",
        "priority_weight": 1.3,
        "summary_keywords": ["refund", "invoice", "duplicate charge"],
        "subject_options": [
            "Customer reports duplicate charge",
            "Refund timeline confirmation request",
        ],
        "body_options": [
            "Customer sees duplicate charge on invoice and asks for correction timeline.",
            "Account manager requests status update for pending reimbursement case.",
        ],
        "sender_options": ["care-escalations@acme-enterprise.com", "finance-ops@acme-enterprise.com"],
    },
    "safety_smoke": {
        "label": "urgent",
        "route_to": "safety",
        "priority_weight": 1.9,
        "summary_keywords": ["safety", "smoke", "overheating"],
        "subject_options": [
            "Safety escalation: smoke seen during charging",
            "Urgent product safety complaint",
        ],
        "body_options": [
            "Customer reports visible smoke and overheating during normal charging operation.",
            "Field ops flagged possible thermal event requiring immediate safety review.",
        ],
        "sender_options": ["support-lead@acme-enterprise.com", "field-ops@acme-enterprise.com"],
    },
    "support_access": {
        "label": "urgent",
        "route_to": "support",
        "priority_weight": 1.4,
        "summary_keywords": ["locked out", "access", "mfa"],
        "subject_options": [
            "Executive locked out after MFA reset",
            "Cannot access admin dashboard before meeting",
        ],
        "body_options": [
            "User cannot access dashboard after security reset and requests immediate help.",
            "Critical user lockout blocks live customer session starting shortly.",
        ],
        "sender_options": ["sales-lead@acme-enterprise.com", "exec-assistant@acme-enterprise.com"],
    },
    "archive_digest": {
        "label": "archive",
        "route_to": "general",
        "priority_weight": 0.7,
        "summary_keywords": ["newsletter", "digest", "no action"],
        "subject_options": [
            "Monthly partner digest",
            "Internal culture newsletter",
        ],
        "body_options": [
            "Sharing monthly digest for awareness only; no action required.",
            "Newsletter update with optional reads and no operational request.",
        ],
        "sender_options": ["people-ops@acme-enterprise.com", "updates@partner-network.io"],
    },
}

PRODUCTION_EVENT_PLAN: list[str] = [
    "phishing_link",
    "incident_checkout",
    "billing_refund",
    "safety_smoke",
    "support_access",
    "archive_digest",
    "incident_checkout",
    "billing_refund",
    "safety_smoke",
    "support_access",
    "archive_digest",
    "incident_checkout",
    "billing_refund",
    "phishing_link",
    "safety_smoke",
    "support_access",
    "incident_checkout",
    "archive_digest",
]

PRODUCTION_PROFILE_EVENT_PLANS: dict[str, list[str]] = {
    "light": PRODUCTION_EVENT_PLAN[:12],
    "standard": PRODUCTION_EVENT_PLAN,
    "heavy": PRODUCTION_EVENT_PLAN
    + [
        "incident_checkout",
        "safety_smoke",
        "incident_checkout",
        "billing_refund",
        "support_access",
        "safety_smoke",
        "phishing_link",
        "incident_checkout",
        "billing_refund",
        "archive_digest",
    ],
}


def _normalize_production_profile(profile_value: object) -> str:
    """Normalize production profile value to one of light/standard/heavy."""
    profile = str(profile_value or "standard").strip().lower()
    return profile if profile in PRODUCTION_PROFILE_EVENT_PLANS else "standard"


def _coerce_bool(value: object, default: bool = False) -> bool:
    """Coerce bool-ish values from runtime options."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _load_private_eval_overrides() -> dict[str, list[dict[str, object]]]:
    """Load private-eval scenarios from OPENENV_PRIVATE_SCENARIOS_JSON.

    Expected shape:
    {
      "task_easy": [ {"scenario_id": "...", "emails": [...], "ground_truth": [...]}, ... ],
      "task_medium": [...],
      "task_hard": [...]
    }
    """
    raw_payload = os.getenv("OPENENV_PRIVATE_SCENARIOS_JSON", "").strip()
    if not raw_payload:
        return {}

    try:
        parsed_payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed_payload, dict):
        return {}

    validated: dict[str, list[dict[str, object]]] = {}
    for task_id, pool in parsed_payload.items():
        if not isinstance(task_id, str) or task_id not in TASK_LIBRARY:
            continue
        if isinstance(pool, list):
            validated[task_id] = cast(list[dict[str, object]], pool)

    return validated


def _build_production_task_definition(
    scenario_index: int,
    split: str,
    runtime_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a deterministic production-style inbox scenario."""
    options = runtime_options or {}
    profile = _normalize_production_profile(options.get("production_profile", "standard"))
    business_hours_mode = _coerce_bool(options.get("business_hours_mode", False), False)

    seed_base = 910000 if split == "private_eval" else 510000
    profile_seed = {"light": 101, "standard": 202, "heavy": 303}[profile]
    rng = random.Random(seed_base + max(0, scenario_index) + profile_seed)

    base_datetime = datetime(2026, 4, 1, 7, 30, tzinfo=timezone.utc)
    base_datetime += timedelta(minutes=max(0, scenario_index) % 720)
    if business_hours_mode:
        base_datetime = base_datetime.replace(hour=9, minute=0)

    thread_counts: dict[str, int] = {}
    emails: list[dict[str, object]] = []
    ground_truth: list[dict[str, object]] = []
    event_plan = PRODUCTION_PROFILE_EVENT_PLANS[profile]

    for idx, template_key in enumerate(event_plan):
        template = PRODUCTION_TEMPLATE_LIBRARY[template_key]
        thread_family = template_key.split("_")[0]
        thread_counts[thread_family] = thread_counts.get(thread_family, 0) + 1
        thread_number = thread_counts[thread_family]

        subject = str(rng.choice(cast(list[str], template["subject_options"])))
        body = str(rng.choice(cast(list[str], template["body_options"])))
        sender = str(rng.choice(cast(list[str], template["sender_options"])))

        if thread_number > 1:
            subject = f"RE[{thread_number}]: {subject}"

        if business_hours_mode:
            business_window_minutes = 8 * 60
            minute_offset = (idx * 17) % business_window_minutes
            day_offset = (idx * 17) // business_window_minutes
            event_dt = (base_datetime + timedelta(days=day_offset, minutes=minute_offset)).replace(
                hour=9 + ((minute_offset // 60) % 8),
                minute=minute_offset % 60,
            )
        else:
            event_dt = base_datetime + timedelta(minutes=idx * 9)
        timestamp = event_dt.isoformat().replace("+00:00", "Z")
        thread_history = []
        if thread_number > 1:
            thread_history.append(
                f"Previous {thread_family} thread update #{thread_number - 1} in operations inbox."
            )

        emails.append(
            {
                "email_id": f"prod-{scenario_index:04d}-{idx + 1:03d}",
                "subject": subject,
                "body": body,
                "sender": sender,
                "timestamp": timestamp,
                "thread_history": thread_history,
            }
        )

        ground_truth.append(
            {
                "label": str(template["label"]),
                "route_to": str(template["route_to"]),
                "priority_weight": float(template["priority_weight"]),
                "summary_keywords": cast(list[str], template["summary_keywords"]),
            }
        )

    return {
        "task_id": "task_production",
        "split": split,
        "scenario_id": f"production-{split}-{profile}-{scenario_index:04d}",
        "description": str(TASK_LIBRARY["task_production"]["description"]),
        "emails": emails,
        "ground_truth": ground_truth,
        "runtime_options": {
            "production_profile": profile,
            "business_hours_mode": business_hours_mode,
        },
    }


def _normalize_split(split: str | None) -> str:
    """Return normalized split name constrained to known values."""
    return "private_eval" if split == "private_eval" else "public"


def get_task_definition(
    task_id: str,
    scenario_index: int = 0,
    split: str | None = None,
    runtime_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return selected task scenario by task_id, split, and deterministic index.

    Args:
        task_id: Task identifier.
        scenario_index: Deterministic index into selected scenario pool.
        split: Scenario split selector; supports public and private_eval.

    Returns:
        Concrete task definition dictionary.

    Raises:
        KeyError: If task_id is not defined.
    """
    if task_id not in TASK_LIBRARY:
        raise KeyError(f"Unknown task_id: {task_id}")

    normalized_split = _normalize_split(split)
    if task_id == "task_production":
        return _build_production_task_definition(
            scenario_index,
            normalized_split,
            runtime_options,
        )

    task_record = TASK_LIBRARY[task_id]
    if normalized_split == "private_eval":
        private_overrides = _load_private_eval_overrides()
        pool = private_overrides.get(task_id, [])
        if not pool:
            pool = cast(list[dict[str, object]], task_record.get("private_eval_pool", []))
        if not pool:
            raise KeyError(
                "No private_eval scenarios configured for "
                f"{task_id}. Set OPENENV_PRIVATE_SCENARIOS_JSON."
            )
    else:
        pool = cast(list[dict[str, object]], task_record.get("scenario_pool", []))

    if not pool:
        raise KeyError(f"No public scenarios configured for {task_id}")

    safe_index = scenario_index % len(pool)
    scenario = pool[safe_index]

    return {
        "task_id": task_id,
        "split": normalized_split,
        "scenario_id": str(scenario.get("scenario_id", f"{task_id}-{safe_index}")),
        "description": str(task_record.get("description", "")),
        "emails": cast(list[dict[str, object]], scenario.get("emails", [])),
        "ground_truth": cast(list[dict[str, object]], scenario.get("ground_truth", [])),
    }


def get_task_scenario_count(task_id: str, split: str | None = None) -> int:
    """Return number of scenarios for a task in selected split."""
    if task_id not in TASK_LIBRARY:
        raise KeyError(f"Unknown task_id: {task_id}")

    if task_id == "task_production":
        return PRODUCTION_SCENARIO_COUNT

    task_record = TASK_LIBRARY[task_id]
    normalized_split = _normalize_split(split)
    if normalized_split == "private_eval":
        private_overrides = _load_private_eval_overrides()
        pool = private_overrides.get(task_id, [])
        if not pool:
            pool = cast(list[dict[str, object]], task_record.get("private_eval_pool", []))
        return len(pool)
    else:
        pool = cast(list[dict[str, object]], task_record.get("scenario_pool", []))

    return len(pool)


def list_task_ids() -> list[str]:
    """Return all supported task identifiers in deterministic order."""
    return ["task_easy", "task_medium", "task_hard", "task_production"]
