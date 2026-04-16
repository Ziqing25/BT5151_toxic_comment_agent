---
name: draft-warning
description: >
  Generate a user-facing moderation warning message using OpenAI gpt-4o-mini,
  or fall back to a deterministic template. Runs after recommend-moderation-action.
  Skipped entirely when action_code is "allow" or "allow_with_monitoring".
applyTo: "draft-warning"
---

## When to Use

Invoke this skill **after** `recommend-moderation-action` has set `action_code` and `severity_label`.
This skill does not change the moderation decision - it only drafts the message that would be sent to
the user. It is skipped when no user notification is required.

---

## How to Execute

1. Read `action_code` from state. If `allow` or `allow_with_monitoring`, write `warning_skipped=True` and an empty `warning_message`; return immediately.
2. Build a prompt using `action_code`, `severity_label`, and `action_label`. **Do not include the raw `comment_text` in the prompt** (privacy).
3. Call OpenAI `gpt-4o-mini` via the `openai` Python SDK. Read the API key from `os.environ["OPENAI_API_KEY"]` - never hardcode it.
4. On any exception (API error, quota, connectivity), fall back to a deterministic template keyed by `action_code`.
5. Write `warning_message`, `warning_skipped`, and `warning_generated_at_utc` to state.

---

## Inputs from Agent State

| Key | Type | Description |
|---|---|---|
| `action_code` | `ActionCode` | Determines skip vs generate |
| `severity_label` | `SeverityLabel` | Informs warning tone and proportionality |
| `action_label` | `str` | Human-readable action name, included in prompt |

---

## Outputs to Agent State

| Key | Type | Description |
|---|---|---|
| `warning_message` | `str` | LLM-generated warning or template fallback; empty string when skipped |
| `warning_skipped` | `bool` | `True` when `action_code` is `allow` or `allow_with_monitoring` |
| `warning_generated_at_utc` | `str` | ISO 8601 UTC timestamp |

---

## Prompt Strategy

The prompt:
- States the severity level and action code
- Asks for a 2-3 sentence professional, non-hostile message addressed to the user
- Explicitly excludes the raw comment text (not sent to OpenAI)
- Requests output that is proportional to severity

Model: `gpt-4o-mini` | `max_tokens=200` | `temperature=0.4`

---

## Fallback Templates

If OpenAI is unavailable, pre-written templates are used per `action_code`:

| Action code | Template summary |
|---|---|
| `soft_warn` | Gentle reminder about community guidelines |
| `review_and_warn` | Comment under review; guidelines reminder |
| `hide_and_review` | Comment hidden pending review; repeated violation warning |
| `remove_and_escalate` | Comment removed; final warning; account restriction notice |

The pipeline **never raises an exception** due to a failed API call - the fallback always fires.

---

## Notes

- This node does not alter `action_code`, `severity_label`, or any upstream decision.
- `warning_message` being an empty string is a valid output (for clean/borderline cases).
- Raw comment text is intentionally excluded from the OpenAI prompt for user privacy.
- API key must be set via `os.environ["OPENAI_API_KEY"]` - in Colab, use `userdata.get("OPENAI_API_KEY")`.
