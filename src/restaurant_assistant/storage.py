"""SQLite storage for web sessions, chat turns and booking records."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from restaurant_assistant.dialogue_state import DialogueState


class BookingStore:
    """Persist lightweight proof-of-concept sessions and booking records."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.initialize()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    closed_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    reference TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    restaurant_name TEXT NOT NULL,
                    food TEXT,
                    area TEXT,
                    pricerange TEXT,
                    address TEXT,
                    postcode TEXT,
                    phone TEXT,
                    day TEXT,
                    booking_date TEXT,
                    time TEXT,
                    people INTEGER,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            self._ensure_session_columns(connection)
            self._ensure_message_columns(connection)

    def create_user(self, username: str, password_hash: str, *, display_name: str | None = None) -> dict[str, Any]:
        normalized = self._normalize_username(username)
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, display_name, password_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, display_name or username.strip(), password_hash, now, now),
            )
            user_id = int(cursor.lastrowid)
        return self.get_user(user_id) or {}

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, display_name, password_hash, created_at, updated_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = self._normalize_username(username)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, display_name, password_hash, created_at, updated_at
                FROM users
                WHERE username = ?
                """,
                (normalized,),
            ).fetchone()
        return dict(row) if row else None

    def ensure_session(self, session_id: str, *, user_id: int | None = None) -> None:
        now = self._now()
        with self._connect() as connection:
            self._ensure_session(connection, session_id, now, user_id=user_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, user_id, created_at, updated_at, status, closed_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def is_session_active(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        return not session or session.get("status") != "closed"

    def close_session(self, session_id: str) -> None:
        self.close_sessions([session_id])

    def close_sessions(self, session_ids: list[str]) -> int:
        cleaned_ids = [session_id for session_id in session_ids if session_id]
        if not cleaned_ids:
            return 0
        now = self._now()
        with self._connect() as connection:
            for session_id in cleaned_ids:
                self._ensure_session(connection, session_id, now)
            connection.execute(
                f"""
                UPDATE sessions
                SET status = 'closed', closed_at = ?, updated_at = ?
                WHERE session_id IN ({",".join("?" for _ in cleaned_ids)})
                """,
                (now, now, *cleaned_ids),
            )
        return len(cleaned_ids)

    def close_all_sessions(self) -> int:
        now = self._now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM sessions WHERE status != 'closed'"
            ).fetchone()
            connection.execute(
                """
                UPDATE sessions
                SET status = 'closed', closed_at = ?, updated_at = ?
                WHERE status != 'closed'
                """,
                (now, now),
            )
        return int(row["count"]) if row else 0

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM bookings WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        metadata: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        now = self._now()
        metadata = metadata or {}
        with self._connect() as connection:
            self._ensure_session(connection, session_id, now)
            connection.execute(
                """
                INSERT INTO messages (
                    session_id, user_message, assistant_message, created_at,
                    intent, effective_intent, slots_json, debug_json, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_message,
                    assistant_message,
                    now,
                    self._nullable_string(metadata.get("intent")),
                    self._nullable_string(metadata.get("effective_intent")),
                    json.dumps(metadata.get("slots") or {}, sort_keys=True),
                    json.dumps(metadata, sort_keys=True, default=str),
                    latency_ms,
                ),
            )

    def list_turns(self, session_id: str, *, include_metadata: bool = False) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user_message, assistant_message, created_at,
                       intent, effective_intent, slots_json, debug_json, latency_ms
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        turns = [dict(row) for row in rows]
        if include_metadata:
            return [self._hydrate_turn_metadata(turn) for turn in turns]
        return [
            {
                "user_message": turn["user_message"],
                "assistant_message": turn["assistant_message"],
                "created_at": turn["created_at"],
            }
            for turn in turns
        ]

    def upsert_booking(self, session_id: str, state: DialogueState) -> None:
        if not state.booking_reference:
            return
        restaurant = state.booking_restaurant or state.selected_restaurant or {}
        name = str(restaurant.get("name") or "the selected restaurant")
        now = self._now()
        with self._connect() as connection:
            self._ensure_session(connection, session_id, now)
            connection.execute(
                """
                INSERT INTO bookings (
                    reference, session_id, restaurant_name, food, area, pricerange,
                    address, postcode, phone, day, booking_date, time, people, status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reference) DO UPDATE SET
                    restaurant_name = excluded.restaurant_name,
                    food = excluded.food,
                    area = excluded.area,
                    pricerange = excluded.pricerange,
                    address = excluded.address,
                    postcode = excluded.postcode,
                    phone = excluded.phone,
                    day = excluded.day,
                    booking_date = excluded.booking_date,
                    time = excluded.time,
                    people = excluded.people,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    state.booking_reference,
                    session_id,
                    name,
                    self._nullable_string(restaurant.get("food")),
                    self._nullable_string(restaurant.get("area")),
                    self._nullable_string(restaurant.get("pricerange")),
                    self._nullable_string(restaurant.get("address")),
                    self._nullable_string(restaurant.get("postcode")),
                    self._nullable_string(restaurant.get("phone")),
                    state.day,
                    state.booking_date,
                    state.time,
                    state.people,
                    state.booking_status,
                    now,
                    now,
                ),
            )

    def list_bookings(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reference, restaurant_name, food, area, pricerange, address,
                       postcode, phone, day, booking_date, time, people, status,
                       created_at, updated_at
                FROM bookings
                WHERE session_id = ?
                ORDER BY updated_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_booking(self, session_id: str, reference: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT reference, restaurant_name, food, area, pricerange, address,
                       postcode, phone, day, booking_date, time, people, status,
                       created_at, updated_at
                FROM bookings
                WHERE session_id = ? AND reference = ?
                """,
                (session_id, reference),
            ).fetchone()
        return dict(row) if row else None

    def get_user_booking(self, user_id: int, reference: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT bookings.reference, bookings.session_id, sessions.user_id,
                       bookings.restaurant_name, bookings.food, bookings.area,
                       bookings.pricerange, bookings.address, bookings.postcode,
                       bookings.phone, bookings.day, bookings.booking_date,
                       bookings.time, bookings.people, bookings.status,
                       bookings.created_at, bookings.updated_at
                FROM bookings
                JOIN sessions ON sessions.session_id = bookings.session_id
                WHERE sessions.user_id = ? AND bookings.reference = ?
                """,
                (user_id, reference),
            ).fetchone()
        return dict(row) if row else None

    def list_all_bookings(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT bookings.reference, bookings.session_id, sessions.user_id,
                       users.username, users.display_name,
                       bookings.restaurant_name, bookings.food, bookings.area,
                       bookings.pricerange, bookings.address, bookings.postcode,
                       bookings.phone, bookings.day, bookings.booking_date,
                       bookings.time, bookings.people, bookings.status,
                       bookings.created_at, bookings.updated_at
                FROM bookings
                LEFT JOIN sessions ON sessions.session_id = bookings.session_id
                LEFT JOIN users ON users.id = sessions.user_id
                ORDER BY bookings.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_all_turns(self, *, include_metadata: bool = True) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, user_message, assistant_message, created_at,
                       intent, effective_intent, slots_json, debug_json, latency_ms
                FROM messages
                ORDER BY id ASC
                """
            ).fetchall()
        turns = [dict(row) for row in rows]
        if include_metadata:
            return [self._hydrate_turn_metadata(turn) for turn in turns]
        return turns

    def list_session_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            sessions = [dict(row) for row in connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()]
        turns = self.list_all_turns(include_metadata=False)
        bookings = self.list_all_bookings()
        users_by_id = {
            user["id"]: user
            for user in self.list_users()
        }
        turns_by_session = Counter(turn["session_id"] for turn in turns)
        bookings_by_session = Counter(booking["session_id"] for booking in bookings)
        confirmed_by_session = Counter(
            booking["session_id"] for booking in bookings if booking.get("status") == "confirmed"
        )
        last_turn_by_session: dict[str, str] = {}
        for turn in turns:
            last_turn_by_session[turn["session_id"]] = turn["created_at"]
        for session in sessions:
            session_id = session["session_id"]
            session["turn_count"] = turns_by_session[session_id]
            session["booking_count"] = bookings_by_session[session_id]
            session["confirmed_booking_count"] = confirmed_by_session[session_id]
            session["last_turn_at"] = last_turn_by_session.get(session_id)
            user = users_by_id.get(session.get("user_id"))
            session["username"] = user.get("username") if user else None
            session["display_name"] = user.get("display_name") if user else None
        return sessions

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, username, display_name, created_at, updated_at
                FROM users
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_user_sessions(self, user_id: int) -> list[dict[str, Any]]:
        sessions = [
            session for session in self.list_session_summaries()
            if session.get("user_id") == user_id
        ]
        return sessions

    def list_user_bookings(self, user_id: int) -> list[dict[str, Any]]:
        session_ids = {session["session_id"] for session in self.list_user_sessions(user_id)}
        return [
            booking for booking in self.list_all_bookings()
            if booking.get("session_id") in session_ids
        ]

    def cancel_user_bookings_except_restaurant(self, user_id: int, keep_restaurant: str) -> list[dict[str, Any]]:
        keep_text = self._normalize_match_text(keep_restaurant)
        cancellable = [
            booking
            for booking in self.list_user_bookings(user_id)
            if booking.get("status") == "confirmed"
            and not self._restaurant_matches_keep_text(str(booking.get("restaurant_name") or ""), keep_text)
        ]
        references = [booking["reference"] for booking in cancellable if booking.get("reference")]
        if not references:
            return []

        now = self._now()
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE bookings
                SET status = 'cancelled', updated_at = ?
                WHERE reference IN ({",".join("?" for _ in references)})
                """,
                (now, *references),
            )
        return [
            {**booking, "status": "cancelled", "updated_at": now}
            for booking in cancellable
        ]

    def kept_user_bookings_for_restaurant(self, user_id: int, keep_restaurant: str) -> list[dict[str, Any]]:
        keep_text = self._normalize_match_text(keep_restaurant)
        return [
            booking
            for booking in self.list_user_bookings(user_id)
            if booking.get("status") == "confirmed"
            and self._restaurant_matches_keep_text(str(booking.get("restaurant_name") or ""), keep_text)
        ]

    def admin_snapshot(self) -> dict[str, Any]:
        sessions = self.list_session_summaries()
        turns = self.list_all_turns(include_metadata=True)
        bookings = self.list_all_bookings()
        intent_counts = Counter(turn.get("effective_intent") or turn.get("intent") or "unknown" for turn in turns)
        status_counts = Counter(booking.get("status") or "unknown" for booking in bookings)
        cuisine_counts = Counter(booking.get("food") or "unknown" for booking in bookings)
        booked_restaurant_counts = Counter(booking.get("restaurant_name") or "unknown" for booking in bookings)
        latencies = [float(turn["latency_ms"]) for turn in turns if turn.get("latency_ms") is not None]
        total_sessions = len(sessions)
        active_sessions = sum(1 for session in sessions if session.get("status") != "closed")
        closed_sessions = sum(1 for session in sessions if session.get("status") == "closed")
        total_turns = len(turns)
        total_bookings = len(bookings)
        sessions_with_booking = len({booking["session_id"] for booking in bookings})
        sessions_with_confirmed = len({booking["session_id"] for booking in bookings if booking.get("status") == "confirmed"})
        booking_attempts = sum(intent_counts[intent] for intent in ("book", "reschedule", "cancel"))
        fallback_count = sum("I can only help" in turn.get("assistant_message", "") for turn in turns)
        clarification_count = sum(
            any(phrase in turn.get("assistant_message", "").lower() for phrase in ("please provide", "which restaurant", "tell me"))
            for turn in turns
        )
        limitation_count = sum(
            any(phrase in turn.get("assistant_message", "").lower() for phrase in ("do not have", "could not find", "not a direct cuisine label"))
            for turn in turns
        )
        return {
            "summary": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "closed_sessions": closed_sessions,
                "total_turns": total_turns,
                "total_bookings": total_bookings,
                "confirmed_bookings": status_counts["confirmed"],
                "cancelled_bookings": status_counts["cancelled"],
                "sessions_with_booking": sessions_with_booking,
                "sessions_with_confirmed_booking": sessions_with_confirmed,
                "booking_conversion_rate": self._rate(sessions_with_confirmed, total_sessions),
                "booking_completion_rate": self._rate(status_counts["confirmed"], booking_attempts),
                "average_turns_per_session": round(total_turns / total_sessions, 2) if total_sessions else 0,
                "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "fallback_count": fallback_count,
                "clarification_count": clarification_count,
                "limitation_count": limitation_count,
            },
            "intent_counts": intent_counts.most_common(),
            "booking_status_counts": status_counts.most_common(),
            "cuisine_counts": cuisine_counts.most_common(8),
            "booked_restaurant_counts": booked_restaurant_counts.most_common(8),
            "sessions": sessions,
            "bookings": bookings,
        }

    def clear_session_messages(self, session_id: str) -> None:
        now = self._now()
        with self._connect() as connection:
            self._ensure_session(connection, session_id, now)
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    def export_session(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": self.list_turns(session_id),
            "bookings": self.list_bookings(session_id),
        }

    def export_session_text(self, session_id: str) -> str:
        payload = self.export_session(session_id)
        lines = [f"Session: {payload['session_id']}"]

        bookings = payload["bookings"]
        if bookings:
            lines.append("")
            lines.append("Bookings:")
            for booking in bookings:
                date_text = booking.get("booking_date") or booking.get("day") or "selected day"
                lines.append(
                    "- "
                    f"{booking.get('reference')}: {booking.get('restaurant_name')} | "
                    f"{date_text} at {booking.get('time')} for {booking.get('people')} people | "
                    f"status: {booking.get('status')}"
                )

        lines.append("")
        lines.append("Conversation:")
        messages = payload["messages"]
        if not messages:
            lines.append("(No messages yet.)")
        for turn in messages:
            lines.append(f"You: {turn['user_message']}")
            lines.append(f"Assistant: {turn['assistant_message']}")

        return "\n".join(lines)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_message_columns(self, connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        columns = {
            "intent": "TEXT",
            "effective_intent": "TEXT",
            "slots_json": "TEXT",
            "debug_json": "TEXT",
            "latency_ms": "REAL",
        }
        for name, column_type in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE messages ADD COLUMN {name} {column_type}")

    def _ensure_session_columns(self, connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        columns = {
            "user_id": "INTEGER",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "closed_at": "TEXT",
        }
        for name, column_type in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} {column_type}")

    def _ensure_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        now: str,
        *,
        user_id: int | None = None,
    ) -> None:
        if user_id is None:
            connection.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at, status)
                VALUES (?, ?, ?, 'active')
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, now, now),
            )
            return
        connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(session_id) DO UPDATE SET
                user_id = excluded.user_id,
                updated_at = excluded.updated_at
            """,
            (session_id, user_id, now, now),
        )

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _nullable_string(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _normalize_username(self, username: str) -> str:
        return username.strip().lower()

    def _normalize_match_text(self, value: str) -> str:
        normalized = str(value).strip().lower()
        normalized = normalized.replace("&", " and ")
        normalized = "".join(character if character.isalnum() else " " for character in normalized)
        return " ".join(normalized.split())

    def _restaurant_matches_keep_text(self, restaurant_name: str, keep_text: str) -> bool:
        normalized_name = self._normalize_match_text(restaurant_name)
        if not keep_text or not normalized_name:
            return False
        return keep_text in normalized_name or normalized_name in keep_text

    def _hydrate_turn_metadata(self, turn: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(turn)
        hydrated["slots"] = self._load_json(hydrated.pop("slots_json", None), default={})
        hydrated["debug"] = self._load_json(hydrated.pop("debug_json", None), default={})
        return hydrated

    def _load_json(self, value: str | None, *, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _rate(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round((numerator / denominator) * 100, 1)

    def to_json(self, session_id: str) -> str:
        return json.dumps(self.export_session(session_id), indent=2, default=str)
