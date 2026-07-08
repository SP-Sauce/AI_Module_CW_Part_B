from dataclasses import replace

from restaurant_assistant.config import get_settings
from restaurant_assistant.web_app import create_app


def register(client, username="demo-user"):
    return client.post(
        "/register",
        data={"username": username, "password": "password123", "display_name": "Demo User"},
        follow_redirects=False,
    )


def admin_login(client, username="admin", password="pass123"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )


def test_web_app_chat_creates_session_booking_record(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    session_response = client.get("/api/session")
    assert session_response.status_code == 200
    session_id = session_response.get_json()["session_id"]
    assert session_id
    assert "greeting" in session_response.get_json()
    assert session_response.get_json()["account"]["username"] == "demo-user"

    search_response = client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    assert search_response.status_code == 200
    assert "Pizza Hut Cherry Hinton" in search_response.get_json()["response"]

    booking_response = client.post("/api/chat", json={"message": "book it for tomorrow at 7pm for 2 people"})
    data = booking_response.get_json()

    assert booking_response.status_code == 200
    assert data["session_id"] == session_id
    assert "Your reference" in data["response"]
    assert len(data["bookings"]) == 1
    assert data["bookings"][0]["restaurant_name"] == "Pizza Hut Cherry Hinton"
    assert data["bookings"][0]["status"] == "confirmed"

    export_response = client.get("/api/export")
    export_data = export_response.get_json()

    assert export_response.status_code == 200
    assert export_data["session_id"] == session_id
    assert "You: I need a cheap Italian restaurant in the south" in export_data["transcript"]
    assert "Assistant:" in export_data["transcript"]
    assert data["bookings"][0]["reference"] in export_data["transcript"]
    assert export_data["messages"][0]["debug"]["effective_intent"] == "search"

    text_download = client.get(f"/api/session/{session_id}/export.txt")
    json_download = client.get(f"/api/session/{session_id}/export.json")

    assert text_download.status_code == 200
    assert text_download.mimetype == "text/plain"
    assert "conversation-" in text_download.headers["Content-Disposition"]
    assert "You: I need a cheap Italian restaurant in the south" in text_download.get_data(as_text=True)
    assert json_download.status_code == 200
    assert json_download.mimetype == "application/json"
    assert "conversation-" in json_download.headers["Content-Disposition"]
    assert '"effective_intent": "search"' in json_download.get_data(as_text=True)


def test_web_app_debug_turns_prints_turn_json_to_terminal(tmp_path, capsys):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True, debug_turns=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    response = client.post("/api/chat", json={"message": "hello"})
    printed = capsys.readouterr().out

    assert response.status_code == 200
    assert '"user_message": "hello"' in printed
    assert '"assistant_message"' in printed
    assert '"debug"' in printed


def test_web_app_new_session_returns_empty_context(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    first = client.get("/api/session").get_json()["session_id"]
    new_session = client.post("/api/new-session").get_json()

    assert new_session["session_id"] != first
    assert new_session["messages"] == []
    assert new_session["bookings"] == []
    assert "Hello" in new_session["greeting"]


def test_web_app_can_open_user_conversation_history(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    first = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south", "session_id": first})
    second = client.post("/api/new-session").get_json()["session_id"]

    opened = client.post(f"/api/session/{first}")
    opened_payload = opened.get_json()
    continued = client.post("/api/chat", json={"message": "thank you", "session_id": first}).get_json()

    assert opened.status_code == 200
    assert opened_payload["session_id"] == first
    assert opened_payload["messages"][0]["user_message"] == "I need a cheap Italian restaurant in the south"
    assert opened_payload["history"]["sessions"][0]["is_current"] is True
    assert continued["session_id"] == first
    assert continued["session_id"] != second


def test_account_history_links_open_saved_conversation(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    first = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south", "session_id": first})
    client.post("/api/new-session")

    history_page = client.get("/history")
    open_response = client.get(f"/session/{first}")
    opened = client.get("/api/session").get_json()

    assert history_page.status_code == 200
    assert f"/session/{first}".encode() in history_page.data
    assert open_response.status_code == 302
    assert open_response.headers["Location"] == "/"
    assert opened["session_id"] == first
    assert opened["messages"][0]["user_message"] == "I need a cheap Italian restaurant in the south"


def test_web_app_bookings_are_account_level_across_chats(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    first = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south", "session_id": first})
    booking_response = client.post(
        "/api/chat",
        json={"message": "book it for tomorrow at 7pm for 2 people", "session_id": first},
    ).get_json()
    reference = booking_response["bookings"][0]["reference"]
    second = client.post("/api/new-session").get_json()
    listed = client.post(
        "/api/chat",
        json={"message": "can you list all bookings", "session_id": second["session_id"]},
    ).get_json()

    assert len(second["bookings"]) == 1
    assert second["bookings"][0]["reference"] == reference
    assert "Current account booking records" in listed["response"]
    assert reference in listed["response"]


def test_web_app_admin_dashboard_reports_metrics(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    client.get("/api/session")
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    client.post("/api/chat", json={"message": "book it for tomorrow at 7pm for 2 people"})
    admin_login(client)

    admin_page = client.get("/admin")
    admin_json = client.get("/api/admin")
    payload = admin_json.get_json()

    assert admin_page.status_code == 200
    assert b"Admin Dashboard" in admin_page.data
    assert b"Intent Distribution" in admin_page.data
    assert b"Logout admin" in admin_page.data
    assert b"Close selected" in admin_page.data
    assert b"Close all" in admin_page.data
    assert admin_json.status_code == 200
    assert payload["summary"]["total_turns"] == 2
    assert payload["summary"]["total_bookings"] == 1
    assert payload["summary"]["active_sessions"] == 1
    assert payload["summary"]["closed_sessions"] == 0
    assert payload["sessions"][0]["messages"][0]["intent"] == "search"
    assert payload["bookings"][0]["username"] == "demo-user"


def test_web_app_end_session_deletes_current_conversation(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    session_id = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    client.post("/api/chat", json={"message": "book it for tomorrow at 7pm for 2 people"})

    ended = client.post("/api/end-session")
    payload = ended.get_json()
    admin_client = app.test_client()
    admin_login(admin_client)
    admin_payload = admin_client.get("/api/admin").get_json()

    assert ended.status_code == 200
    assert payload["session_id"] != session_id
    assert payload["messages"] == []
    assert payload["bookings"] == []
    assert "Hello" in payload["greeting"]
    assert session_id not in {session["session_id"] for session in admin_payload["sessions"]}


def test_web_app_admin_can_close_session_and_force_new_chat_session(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    session_id = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    admin_client = app.test_client()
    admin_login(admin_client)

    close_response = admin_client.post(f"/admin/sessions/{session_id}/close")
    admin_payload = admin_client.get("/api/admin").get_json()
    next_session = client.get("/api/session").get_json()

    closed_session = next(session for session in admin_payload["sessions"] if session["session_id"] == session_id)
    assert close_response.status_code == 302
    assert closed_session["status"] == "closed"
    assert admin_payload["summary"]["closed_sessions"] == 1
    assert next_session["session_id"] != session_id
    assert next_session["messages"] == []


def test_web_app_admin_can_close_selected_and_all_sessions(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()
    register(client)

    first = client.get("/api/session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    second = client.post("/api/new-session").get_json()["session_id"]
    client.post("/api/chat", json={"message": "Show me Turkish restaurants"})
    admin_client = app.test_client()
    admin_login(admin_client)

    selected_response = admin_client.post("/admin/sessions/close-selected", data={"session_ids": [first]})
    selected_payload = admin_client.get("/api/admin").get_json()
    first_session = next(session for session in selected_payload["sessions"] if session["session_id"] == first)
    second_session = next(session for session in selected_payload["sessions"] if session["session_id"] == second)

    assert selected_response.status_code == 302
    assert first_session["status"] == "closed"
    assert second_session["status"] == "active"
    assert selected_payload["summary"]["active_sessions"] == 1
    assert selected_payload["summary"]["closed_sessions"] == 1

    all_response = admin_client.post("/admin/sessions/close-all")
    all_payload = admin_client.get("/api/admin").get_json()

    assert all_response.status_code == 302
    assert all_payload["summary"]["active_sessions"] == 0
    assert all_payload["summary"]["closed_sessions"] == 2


def test_web_app_requires_login_for_chat_api(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.get("/api/session")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Login required."


def test_web_app_requires_admin_login_for_admin_pages_and_api(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)
    client = app.test_client()

    page = client.get("/admin")
    api = client.get("/api/admin")
    bad_login = admin_login(client, password="wrong")
    good_login = admin_login(client)
    authed_page = client.get("/admin")

    assert page.status_code == 302
    assert page.headers["Location"] == "/login"
    assert api.status_code == 401
    assert api.get_json()["error"] == "Admin login required."
    assert bad_login.status_code == 400
    assert good_login.status_code == 302
    assert authed_page.status_code == 200
    assert b"Admin Dashboard" in authed_page.data


def test_web_app_shared_login_routes_admin_and_user(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)

    admin_client = app.test_client()
    admin_response = admin_login(admin_client)
    admin_page = admin_client.get("/admin")
    admin_chat_api = admin_client.get("/api/session")

    user_client = app.test_client()
    register(user_client, username="shared-user")
    user_login_page = user_client.get("/")

    assert admin_response.status_code == 302
    assert admin_response.headers["Location"] == "/admin"
    assert admin_page.status_code == 200
    assert admin_chat_api.status_code == 401
    assert user_login_page.status_code == 200
    assert b"MultiWOZ Restaurant Assistant" in user_login_page.data


def test_web_app_account_history_is_scoped_by_user(tmp_path):
    settings = replace(get_settings(), booking_db_path=tmp_path / "bookings.sqlite3")
    app = create_app(settings=settings, use_sample=True)
    app.config.update(TESTING=True)

    first_client = app.test_client()
    register(first_client, username="first-user")
    first_client.get("/api/session")
    first_client.post("/api/chat", json={"message": "I need a cheap Italian restaurant in the south"})
    first_client.post("/api/chat", json={"message": "book it for tomorrow at 7pm for 2 people"})
    first_history = first_client.get("/api/account/history").get_json()

    second_client = app.test_client()
    register(second_client, username="second-user")
    second_history = second_client.get("/api/account/history").get_json()

    assert len(first_history["sessions"]) == 1
    assert len(first_history["bookings"]) == 1
    assert second_history["sessions"] == []
    assert second_history["bookings"] == []
