"""Flask web interface for the restaurant assistant."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session as flask_session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from restaurant_assistant.assistant import RestaurantAssistant
from restaurant_assistant.config import Settings, get_settings
from restaurant_assistant.data_loader import load_restaurants
from restaurant_assistant.dialogue_state import DialogueState
from restaurant_assistant.storage import BookingStore


SESSION_COOKIE = "restaurant_session_id"
SELECTED_SESSION_KEY = "restaurant_selected_session_id"
SESSION_PATTERN = re.compile(r"^[a-f0-9]{32}$")
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "pass123"
GREETING = (
    "Hello. I can help you find restaurants from the MultiWOZ restaurant records, "
    "create booking records, and manage bookings in this conversation."
)


def create_app(
    *,
    settings: Settings | None = None,
    use_sample: bool = False,
    enable_llm: bool | None = None,
    debug_turns: bool = False,
) -> Flask:
    """Create the local browser app."""

    settings = settings or get_settings()
    if enable_llm is not None:
        settings = replace(settings, enable_llm=enable_llm)

    package_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_dir / "templates"),
        static_folder=str(package_dir / "static"),
    )
    app.config["JSON_SORT_KEYS"] = False
    app.secret_key = "local-restaurant-assistant-dev-key"

    restaurants = load_restaurants(settings, use_sample=use_sample)
    store = BookingStore(settings.booking_db_path)
    assistants: dict[str, RestaurantAssistant] = {}
    lock = Lock()

    def get_current_user() -> dict[str, Any] | None:
        user_id = flask_session.get("user_id")
        if not isinstance(user_id, int):
            return None
        return store.get_user(user_id)

    def is_admin_authenticated() -> bool:
        return flask_session.get("admin_authenticated") is True

    def admin_login_required_response():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Admin login required."}), 401
        return redirect(url_for("login"))

    def get_session_id(user_id: int) -> tuple[str, bool]:
        existing = request.cookies.get(SESSION_COOKIE, "")
        if SESSION_PATTERN.fullmatch(existing) and store.is_session_active(existing):
            existing_session = store.get_session(existing)
            if existing_session and existing_session.get("user_id") == user_id:
                store.ensure_session(existing, user_id=user_id)
                return existing, False
        session_id = uuid.uuid4().hex
        store.ensure_session(session_id, user_id=user_id)
        return session_id, True

    def get_owned_session(session_id: str, user_id: int) -> dict[str, Any] | None:
        if not SESSION_PATTERN.fullmatch(session_id):
            return None
        session = store.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return None
        return session

    def get_selected_session(user_id: int) -> tuple[str | None, dict[str, Any] | None]:
        selected = flask_session.get(SELECTED_SESSION_KEY)
        if not isinstance(selected, str):
            return None, None
        session = get_owned_session(selected, user_id)
        if not session:
            flask_session.pop(SELECTED_SESSION_KEY, None)
            return None, None
        return selected, session

    def get_chat_session_id(user_id: int, requested_session_id: str | None) -> tuple[str, bool]:
        if requested_session_id:
            session = get_owned_session(requested_session_id, user_id)
            if session and session.get("status") != "closed":
                store.ensure_session(requested_session_id, user_id=user_id)
                return requested_session_id, False
        return get_session_id(user_id)

    def get_assistant(session_id: str, *, user_id: int) -> RestaurantAssistant:
        with lock:
            assistant = assistants.get(session_id)
            if assistant is None:
                assistant = RestaurantAssistant(
                    restaurants=restaurants,
                    settings=settings,
                    enable_llm=settings.enable_llm,
                    booking_store=store,
                    session_id=session_id,
                    user_id=user_id,
                )
                hydrate_state(assistant.state, store, session_id)
                assistants[session_id] = assistant
            return assistant

    def json_with_cookie(payload: dict[str, Any], session_id: str, is_new: bool):
        response = jsonify(payload)
        if is_new:
            response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return response

    @app.get("/")
    def index():
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        selected_session_id, _ = get_selected_session(user["id"])
        if selected_session_id:
            session_id, is_new = selected_session_id, False
        else:
            session_id, is_new = get_session_id(user["id"])
        response = make_response(render_template("index.html", session_id=session_id, user=user))
        if is_new:
            response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return response

    @app.get("/login")
    def login():
        if is_admin_authenticated():
            return redirect(url_for("admin_dashboard"))
        if get_current_user():
            return redirect(url_for("index"))
        return render_template("auth.html", mode="login", error=None)

    @app.post("/login")
    def login_post():
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))
        if username.lower() == ADMIN_USERNAME:
            if password != ADMIN_PASSWORD:
                return render_template("auth.html", mode="login", error="Invalid username or password."), 400
            flask_session.pop("user_id", None)
            flask_session["admin_authenticated"] = True
            response = redirect(url_for("admin_dashboard"))
            response.delete_cookie(SESSION_COOKIE)
            return response
        user = store.get_user_by_username(username)
        if not user or not check_password_hash(user["password_hash"], password):
            return render_template("auth.html", mode="login", error="Invalid username or password."), 400
        flask_session.pop("admin_authenticated", None)
        flask_session["user_id"] = user["id"]
        return redirect(url_for("index"))

    @app.get("/register")
    def register():
        if is_admin_authenticated():
            return redirect(url_for("admin_dashboard"))
        if get_current_user():
            return redirect(url_for("index"))
        return render_template("auth.html", mode="register", error=None)

    @app.post("/register")
    def register_post():
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))
        display_name = str(request.form.get("display_name", "")).strip() or username
        if len(username) < 3:
            return render_template("auth.html", mode="register", error="Username must be at least 3 characters."), 400
        if len(password) < 6:
            return render_template("auth.html", mode="register", error="Password must be at least 6 characters."), 400
        try:
            user = store.create_user(username, generate_password_hash(password), display_name=display_name)
        except sqlite3.IntegrityError:
            return render_template("auth.html", mode="register", error="That username is already registered."), 400
        flask_session.pop("admin_authenticated", None)
        flask_session["user_id"] = user["id"]
        return redirect(url_for("index"))

    @app.post("/logout")
    def logout():
        flask_session.pop("user_id", None)
        flask_session.pop("admin_authenticated", None)
        response = redirect(url_for("login"))
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.get("/history")
    def account_history():
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        return render_template("history.html", user=user, history=account_history_payload(store, user["id"]))

    @app.get("/session/<session_id>")
    def open_session_page(session_id: str):
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        session = get_owned_session(session_id, user["id"])
        if not session:
            return redirect(url_for("account_history"))

        response = redirect(url_for("index"))
        if session.get("status") == "closed":
            flask_session[SELECTED_SESSION_KEY] = session_id
            return response

        flask_session.pop(SELECTED_SESSION_KEY, None)
        store.ensure_session(session_id, user_id=user["id"])
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return response

    @app.get("/api/session")
    def session_snapshot():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        selected_session_id = flask_session.pop(SELECTED_SESSION_KEY, None)
        if isinstance(selected_session_id, str):
            selected_session = get_owned_session(selected_session_id, user["id"])
            if selected_session:
                get_assistant(selected_session_id, user_id=user["id"])
                return jsonify(session_payload(store, selected_session_id, user=user))
        session_id, is_new = get_session_id(user["id"])
        get_assistant(session_id, user_id=user["id"])
        return json_with_cookie(session_payload(store, session_id, user=user), session_id, is_new)

    @app.post("/api/session/<session_id>")
    def open_session(session_id: str):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session = get_owned_session(session_id, user["id"])
        if not session:
            return jsonify({"error": "Conversation not found for this account."}), 404
        if session.get("status") != "closed":
            store.ensure_session(session_id, user_id=user["id"])
            session = store.get_session(session_id) or session
        get_assistant(session_id, user_id=user["id"])
        response = jsonify(session_payload(store, session_id, user=user))
        if session.get("status") != "closed":
            response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return response

    @app.get("/api/export")
    def export_session():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session_id, is_new = get_session_id(user["id"])
        get_assistant(session_id, user_id=user["id"])
        return json_with_cookie(
            session_export_payload(store, session_id, user=user),
            session_id,
            is_new,
        )

    @app.get("/api/session/<session_id>/export")
    def export_named_session(session_id: str):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session = get_owned_session(session_id, user["id"])
        if not session:
            return jsonify({"error": "Conversation not found for this account."}), 404
        return jsonify(session_export_payload(store, session_id, user=user))

    @app.get("/api/session/<session_id>/export.txt")
    def export_named_session_text(session_id: str):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session = get_owned_session(session_id, user["id"])
        if not session:
            return jsonify({"error": "Conversation not found for this account."}), 404
        return text_export_response(store.export_session_text(session_id), session_id)

    @app.get("/api/session/<session_id>/export.json")
    def export_named_session_json(session_id: str):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session = get_owned_session(session_id, user["id"])
        if not session:
            return jsonify({"error": "Conversation not found for this account."}), 404
        return json_export_response(session_export_payload(store, session_id, user=user), session_id)

    @app.post("/api/chat")
    def chat():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        payload = request.get_json(silent=True) or {}
        requested_session_id = str(payload.get("session_id", "")).strip()
        session_id, is_new = get_chat_session_id(user["id"], requested_session_id)
        message = str(payload.get("message", "")).strip()
        if not message:
            return json_with_cookie({"error": "Message is required.", "session_id": session_id}, session_id, is_new), 400
        assistant = get_assistant(session_id, user_id=user["id"])
        started = perf_counter()
        result = assistant.process(message, debug=True)
        latency_ms = round((perf_counter() - started) * 1000, 1)
        turn_metadata = build_turn_metadata(result.debug)
        store.save_turn(
            session_id,
            message,
            result.response,
            metadata=turn_metadata,
            latency_ms=latency_ms,
        )
        if debug_turns:
            print(
                json.dumps(
                    {
                        "session_id": session_id,
                        "user_message": message,
                        "assistant_message": result.response,
                        "latency_ms": latency_ms,
                        "debug": result.debug,
                    },
                    indent=2,
                    default=str,
                ),
                flush=True,
            )
        return json_with_cookie(
            {
                "session_id": session_id,
                "session_status": (store.get_session(session_id) or {}).get("status", "active"),
                "account": public_user(user),
                "response": result.response,
                "bookings": store.list_user_bookings(user["id"]),
                "history": account_history_payload(store, user["id"], current_session_id=session_id),
                "export": session_export_payload(store, session_id, user=user),
            },
            session_id,
            is_new,
        )

    @app.post("/api/new-session")
    def new_session():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session_id = uuid.uuid4().hex
        store.ensure_session(session_id, user_id=user["id"])
        with lock:
            assistants[session_id] = RestaurantAssistant(
                restaurants=restaurants,
                settings=settings,
                enable_llm=settings.enable_llm,
                booking_store=store,
                session_id=session_id,
                user_id=user["id"],
            )
        response = jsonify(session_payload(store, session_id, user=user))
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return response

    @app.post("/api/end-session")
    def end_session():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        session_id, _ = get_session_id(user["id"])
        store.delete_session(session_id)
        with lock:
            assistants.pop(session_id, None)

        new_session_id = uuid.uuid4().hex
        store.ensure_session(new_session_id, user_id=user["id"])
        with lock:
            assistants[new_session_id] = RestaurantAssistant(
                restaurants=restaurants,
                settings=settings,
                enable_llm=settings.enable_llm,
                booking_store=store,
                session_id=new_session_id,
                user_id=user["id"],
            )
        response = jsonify(session_payload(store, new_session_id, user=user))
        response.set_cookie(SESSION_COOKIE, new_session_id, httponly=True, samesite="Lax")
        return response

    @app.get("/api/account/history")
    def account_history_api():
        user = get_current_user()
        if not user:
            return jsonify({"error": "Login required."}), 401
        return jsonify(account_history_payload(store, user["id"]))

    @app.get("/admin")
    def admin_dashboard():
        if not is_admin_authenticated():
            return admin_login_required_response()
        return render_template("admin.html", snapshot=admin_snapshot_with_conversations(store))

    @app.get("/admin/login")
    def admin_login():
        return redirect(url_for("login"))

    @app.post("/admin/login")
    def admin_login_post():
        return redirect(url_for("login"), code=307)

    @app.post("/admin/logout")
    def admin_logout():
        return logout()

    @app.get("/api/admin")
    def admin_api():
        if not is_admin_authenticated():
            return admin_login_required_response()
        return jsonify(admin_snapshot_with_conversations(store))

    @app.post("/admin/sessions/<session_id>/close")
    def admin_close_session(session_id: str):
        if not is_admin_authenticated():
            return admin_login_required_response()
        store.close_session(session_id)
        with lock:
            assistants.pop(session_id, None)
        return redirect(url_for("admin_dashboard"))

    @app.post("/admin/sessions/close-selected")
    def admin_close_selected_sessions():
        if not is_admin_authenticated():
            return admin_login_required_response()
        session_ids = valid_session_ids(request.form.getlist("session_ids"))
        store.close_sessions(session_ids)
        with lock:
            for session_id in session_ids:
                assistants.pop(session_id, None)
        return redirect(url_for("admin_dashboard"))

    @app.post("/admin/sessions/close-all")
    def admin_close_all_sessions():
        if not is_admin_authenticated():
            return admin_login_required_response()
        store.close_all_sessions()
        with lock:
            assistants.clear()
        return redirect(url_for("admin_dashboard"))

    @app.post("/api/admin/sessions/<session_id>/close")
    def admin_close_session_api(session_id: str):
        if not is_admin_authenticated():
            return admin_login_required_response()
        store.close_session(session_id)
        with lock:
            assistants.pop(session_id, None)
        return jsonify({"session_id": session_id, "status": "closed"})

    @app.post("/api/admin/sessions/close-selected")
    def admin_close_selected_sessions_api():
        if not is_admin_authenticated():
            return admin_login_required_response()
        payload = request.get_json(silent=True) or {}
        raw_session_ids = payload.get("session_ids", [])
        if not isinstance(raw_session_ids, list):
            raw_session_ids = []
        session_ids = valid_session_ids([str(session_id) for session_id in raw_session_ids])
        closed_count = store.close_sessions(session_ids)
        with lock:
            for session_id in session_ids:
                assistants.pop(session_id, None)
        return jsonify({"closed_count": closed_count, "session_ids": session_ids})

    @app.post("/api/admin/sessions/close-all")
    def admin_close_all_sessions_api():
        if not is_admin_authenticated():
            return admin_login_required_response()
        closed_count = store.close_all_sessions()
        with lock:
            assistants.clear()
        return jsonify({"closed_count": closed_count})

    return app


def build_turn_metadata(debug: dict[str, Any]) -> dict[str, Any]:
    """Keep the stored analytics compact and demo-friendly."""

    return {
        "turn_timestamp": debug.get("turn_timestamp"),
        "intent": debug.get("intent"),
        "effective_intent": debug.get("effective_intent"),
        "slots": debug.get("slots", {}),
        "unsupported_slots": debug.get("unsupported_slots", {}),
        "relative_day_resolution": debug.get("relative_day_resolution"),
        "slot_extraction_attempted_llm": debug.get("slot_extraction_attempted_llm"),
        "slot_extraction_used_llm": debug.get("slot_extraction_used_llm"),
        "slot_extraction_errors": debug.get("slot_extraction_errors", []),
        "slot_model_name": debug.get("slot_model_name"),
        "generation_mode": debug.get("generation_mode"),
        "retrieved_count": len(debug.get("retrieved_restaurants", [])),
        "ranked_count": len(debug.get("ranking", [])),
    }


def session_payload(store: BookingStore, session_id: str, *, user: dict[str, Any]) -> dict[str, Any]:
    session = store.get_session(session_id) or {}
    return {
        "session_id": session_id,
        "session_status": session.get("status", "active"),
        "account": public_user(user),
        "greeting": GREETING,
        "messages": store.list_turns(session_id),
        "bookings": store.list_user_bookings(user["id"]),
        "session_bookings": store.list_bookings(session_id),
        "history": account_history_payload(store, user["id"], current_session_id=session_id),
        "export": session_export_payload(store, session_id, user=user),
    }


def session_export_payload(store: BookingStore, session_id: str, *, user: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "account": public_user(user),
        "transcript": store.export_session_text(session_id),
        "messages": store.list_turns(session_id, include_metadata=True),
        "bookings": store.list_bookings(session_id),
    }


def text_export_response(transcript: str, session_id: str) -> Response:
    return Response(
        transcript,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=conversation-{session_id}.txt"},
    )


def json_export_response(payload: dict[str, Any], session_id: str) -> Response:
    return Response(
        json.dumps(payload, indent=2, default=str),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=conversation-{session_id}.json"},
    )


def valid_session_ids(session_ids: list[str]) -> list[str]:
    return [session_id for session_id in session_ids if SESSION_PATTERN.fullmatch(session_id)]


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
    }


def account_history_payload(
    store: BookingStore,
    user_id: int,
    *,
    current_session_id: str | None = None,
) -> dict[str, Any]:
    sessions = []
    for session in store.list_user_sessions(user_id):
        session_id = session["session_id"]
        if (
            not session.get("turn_count")
            and not session.get("booking_count")
            and session_id != current_session_id
        ):
            continue
        turns = store.list_turns(session_id)
        last_turn = turns[-1] if turns else None
        sessions.append(
            {
                "session_id": session_id,
                "is_current": session_id == current_session_id,
                "status": session.get("status", "active"),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "turn_count": session.get("turn_count", 0),
                "booking_count": session.get("booking_count", 0),
                "last_user_message": last_turn.get("user_message") if last_turn else None,
            }
        )
    bookings = store.list_user_bookings(user_id)
    return {
        "sessions": sessions,
        "bookings": bookings,
    }


def admin_snapshot_with_conversations(store: BookingStore) -> dict[str, Any]:
    snapshot = store.admin_snapshot()
    sessions = []
    for session in snapshot["sessions"]:
        session = dict(session)
        session_id = session["session_id"]
        session["messages"] = store.list_turns(session_id, include_metadata=True)
        session["bookings"] = store.list_bookings(session_id)
        sessions.append(session)
    snapshot["sessions"] = sessions
    return snapshot


def hydrate_state(state: DialogueState, store: BookingStore, session_id: str) -> None:
    """Restore enough state for booking follow-ups after a page refresh or restart."""

    turns = store.list_turns(session_id)
    for turn in turns:
        state.add_turn(turn["user_message"], turn["assistant_message"], timestamp=turn["created_at"])

    active_booking = next(
        (booking for booking in store.list_bookings(session_id) if booking.get("status") == "confirmed"),
        None,
    )
    if not active_booking:
        return

    restaurant = {
        "name": active_booking.get("restaurant_name"),
        "food": active_booking.get("food"),
        "area": active_booking.get("area"),
        "pricerange": active_booking.get("pricerange"),
        "address": active_booking.get("address"),
        "postcode": active_booking.get("postcode"),
        "phone": active_booking.get("phone"),
    }
    state.food = active_booking.get("food")
    state.area = active_booking.get("area")
    state.pricerange = active_booking.get("pricerange")
    state.day = active_booking.get("day")
    state.booking_date = active_booking.get("booking_date")
    state.time = active_booking.get("time")
    state.people = active_booking.get("people")
    state.selected_restaurant = restaurant
    state.booking_restaurant = restaurant
    state.booking_status = "confirmed"
    state.booking_reference = active_booking.get("reference")
