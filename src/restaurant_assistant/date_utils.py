"""Date helpers for resolving relative booking days."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATIVE_DAYS = {"today", "tomorrow", "day_after", "day_after_tomorrow"}
DAY_MODIFIERS = {"this", "next", "next_week"}


def now_in_timezone(timezone: str = "Europe/London") -> datetime:
    """Return the current time in the configured local timezone."""

    return datetime.now(ZoneInfo(timezone))


def resolve_relative_day(
    relative_day: str,
    *,
    reference_time: datetime,
    active_booking_day: str | None = None,
) -> str:
    """Resolve relative day language to a weekday name.

    `today`, `tomorrow` and `day_after_tomorrow` are resolved from the prompt
    timestamp. A standalone `day_after` uses the active booking day when one is
    available, otherwise it falls back to the day after tomorrow from the prompt
    timestamp.
    """

    normalized = relative_day.strip().lower()
    if normalized == "today":
        return WEEKDAYS[reference_time.weekday()]
    if normalized == "tomorrow":
        return WEEKDAYS[(reference_time + timedelta(days=1)).weekday()]
    if normalized == "day_after_tomorrow":
        return WEEKDAYS[(reference_time + timedelta(days=2)).weekday()]
    if normalized == "day_after":
        if active_booking_day in WEEKDAYS:
            return WEEKDAYS[(WEEKDAYS.index(active_booking_day) + 1) % 7]
        return WEEKDAYS[(reference_time + timedelta(days=2)).weekday()]
    return normalized


def resolve_relative_day_date(
    relative_day: str,
    *,
    reference_time: datetime,
    active_booking_date: str | None = None,
) -> date:
    """Resolve relative day language to a concrete calendar date."""

    normalized = relative_day.strip().lower()
    if normalized == "today":
        return reference_time.date()
    if normalized == "tomorrow":
        return reference_time.date() + timedelta(days=1)
    if normalized == "day_after_tomorrow":
        return reference_time.date() + timedelta(days=2)
    if normalized == "day_after":
        if active_booking_date:
            return date.fromisoformat(active_booking_date) + timedelta(days=1)
        return reference_time.date() + timedelta(days=2)
    return reference_time.date()


def resolve_weekday_date(
    day: str,
    *,
    reference_time: datetime,
    modifier: str | None = None,
    active_booking_date: str | None = None,
) -> date:
    """Resolve a weekday mention to a concrete date.

    `next_week` means the named weekday in the next Monday-Sunday calendar week.
    `next` means the next occurrence after today. Bare weekdays use the active
    booking date as an anchor when one is supplied, otherwise they use the next
    occurrence from today, allowing today.
    """

    normalized_day = day.strip().lower()
    if normalized_day not in WEEKDAYS:
        return reference_time.date()

    today = reference_time.date()
    target_weekday = WEEKDAYS.index(normalized_day)
    normalized_modifier = (modifier or "").strip().lower()

    if normalized_modifier == "next_week":
        today_weekday = reference_time.weekday()
        start_of_this_week = today - timedelta(days=today_weekday)
        return start_of_this_week + timedelta(days=7 + target_weekday)

    anchor = today
    if not normalized_modifier and active_booking_date:
        anchor = max(date.fromisoformat(active_booking_date), today)

    days_ahead = (target_weekday - anchor.weekday()) % 7
    if normalized_modifier == "next" and days_ahead == 0:
        days_ahead = 7
    return anchor + timedelta(days=days_ahead)


def format_booking_date(booking_date: str | None, fallback_day: str | None = None) -> str:
    """Format an ISO booking date for responses."""

    if not booking_date:
        return fallback_day or "the selected day"
    parsed = date.fromisoformat(booking_date)
    weekday = WEEKDAYS[parsed.weekday()].capitalize()
    return f"{weekday} {parsed.day} {parsed.strftime('%B %Y')}"
