"""Date helpers for resolving relative booking days."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATIVE_DAYS = {"today", "tomorrow", "day_after", "day_after_tomorrow"}


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

