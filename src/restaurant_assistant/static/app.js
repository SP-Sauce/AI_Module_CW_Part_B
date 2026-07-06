const sessionId = document.getElementById("sessionId");
const messages = document.getElementById("messages");
const bookings = document.getElementById("bookings");
const bookingCount = document.getElementById("bookingCount");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const newSessionButton = document.getElementById("newSessionButton");
const copyHistoryButton = document.getElementById("copyHistoryButton");
const historyList = document.getElementById("historyList");
const historyCount = document.getElementById("historyCount");
const bookingModal = document.getElementById("bookingModal");
const bookingModalTitle = document.getElementById("bookingModalTitle");
const bookingModalDetails = document.getElementById("bookingModalDetails");
const bookingModalCopy = document.getElementById("bookingModalCopy");
const bookingModalCalendar = document.getElementById("bookingModalCalendar");
const bookingModalClose = document.getElementById("bookingModalClose");

let currentSessionId = sessionId?.textContent?.trim() || "";
let currentSessionStatus = "active";
let selectedBooking = null;

function appendMessage(role, text, options = {}) {
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    renderAssistantContent(bubble, text);
  } else {
    bubble.textContent = text;
  }

  row.appendChild(bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function renderAssistantContent(bubble, text) {
  const bookingList = parseBookingList(text);
  if (bookingList) {
    renderBookingTable(bubble, bookingList);
    return;
  }

  const parsed = parseRestaurantList(text);
  if (!parsed) {
    bubble.textContent = text;
    return;
  }

  bubble.classList.add("has-table");
  if (parsed.before) {
    const intro = document.createElement("p");
    intro.className = "assistant-intro";
    intro.textContent = parsed.before;
    bubble.appendChild(intro);
  }

  const title = document.createElement("div");
  title.className = "table-title";
  title.textContent = parsed.title;
  bubble.appendChild(title);

  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";

  const table = document.createElement("table");
  table.className = "restaurant-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Restaurant", "Cuisine", "Price", "Area"].forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  const tbody = document.createElement("tbody");
  parsed.rows.forEach((item) => {
    const tr = document.createElement("tr");
    [item.name, item.food, item.price, item.area].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrapper.appendChild(table);
  bubble.appendChild(wrapper);
}

function renderBookingTable(bubble, parsed) {
  bubble.classList.add("has-table");

  const title = document.createElement("div");
  title.className = "table-title";
  title.textContent = parsed.title;
  bubble.appendChild(title);

  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";

  const table = document.createElement("table");
  table.className = "restaurant-table booking-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Reference", "Restaurant", "Date", "Time", "People", "Status"].forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  const tbody = document.createElement("tbody");
  parsed.rows.forEach((item) => {
    const tr = document.createElement("tr");

    const refCell = document.createElement("td");
    refCell.className = "booking-ref-cell";
    const refText = document.createElement("span");
    refText.textContent = item.reference;
    const copyButton = makeCopyButton(item.reference);
    const calendarButton = makeCalendarButton(item);
    refCell.appendChild(refText);
    refCell.appendChild(copyButton);
    refCell.appendChild(calendarButton);
    tr.appendChild(refCell);

    [item.restaurant, item.date, item.time, item.people, item.status].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrapper.appendChild(table);
  bubble.appendChild(wrapper);
}

function makeCopyButton(value) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-ref-button";
  button.textContent = "Copy";
  button.title = "Copy booking reference";
  button.addEventListener("click", async () => {
    const original = button.textContent;
    button.disabled = true;
    try {
      await copyText(value);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Failed";
    } finally {
      window.setTimeout(() => {
        button.textContent = original;
        button.disabled = false;
      }, 1200);
    }
  });
  return button;
}

function makeCalendarButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-ref-button calendar-button";
  button.textContent = "Calendar";
  button.title = "Download calendar event";
  button.addEventListener("click", () => {
    const event = buildCalendarEvent(item);
    if (!event) {
      button.textContent = "Missing date";
      window.setTimeout(() => {
        button.textContent = "Calendar";
      }, 1200);
      return;
    }
    const filename = `${safeFilename(item.reference || "booking")}.ics`;
    downloadTextFile(filename, event, "text/calendar;charset=utf-8");
  });
  return button;
}

function makeDetailsButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-ref-button details-button";
  button.textContent = "Details";
  button.title = "Open booking details";
  button.addEventListener("click", () => showBookingModal(item));
  return button;
}

function parseRestaurantList(text) {
  const marker = /(Matching restaurants:|Other matching options:)/.exec(text);
  if (!marker) {
    return null;
  }

  const before = text.slice(0, marker.index).trim();
  const title = marker[1].replace(":", "");
  const listText = text.slice(marker.index + marker[1].length).trim();
  const itemPattern =
    /(\d+)\.\s+(.+?)\s+\((cheap|moderate|expensive|unknown price)\s+(.+?),\s+(.+?)\)(?=\s+\d+\.|$)/g;
  const rows = [];
  let match = itemPattern.exec(listText);
  while (match) {
    rows.push({
      name: match[2].trim(),
      price: match[3].trim(),
      food: match[4].trim(),
      area: match[5].trim(),
    });
    match = itemPattern.exec(listText);
  }

  if (!rows.length) {
    return null;
  }
  return { before, title, rows };
}

function parseBookingList(text) {
  const marker = /(Current session booking records:|Current account booking records:)/.exec(text);
  if (!marker) {
    return null;
  }

  const listText = text.slice(marker.index + marker[1].length).trim();
  const itemPattern =
    /(\d+)\.\s+(BK-[A-Z0-9]{6}):\s+(.+?)\s+on\s+(.+?)\s+at\s+(\d{2}:\d{2})\s+for\s+(\d+)\s+people\s+\(([^)]+)\)(?=\s+\d+\.|$)/g;
  const rows = [];
  let match = itemPattern.exec(listText);
  while (match) {
    rows.push({
      reference: match[2].trim(),
      restaurant: match[3].trim(),
      date: match[4].trim(),
      time: match[5].trim(),
      people: match[6].trim(),
      status: match[7].trim(),
    });
    match = itemPattern.exec(listText);
  }

  if (!rows.length) {
    return null;
  }
  return { title: marker[1].replace(":", ""), rows };
}

function buildCalendarEvent(item) {
  const start = getCalendarStart(item);
  if (!start) {
    return null;
  }
  const end = addMinutes(start, 90);
  const restaurant = item.restaurant || item.restaurant_name || "Restaurant booking";
  const reference = item.reference || "booking";
  const people = item.people || "?";
  const status = item.status || "confirmed";
  const location = [item.address, item.postcode].filter(Boolean).join(", ");
  const description = [
    `Booking reference: ${reference}`,
    `Restaurant: ${restaurant}`,
    `People: ${people}`,
    `Status: ${status}`,
    "Generated by the MultiWOZ restaurant assistant.",
  ].join("\n");

  return [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//MultiWOZ Restaurant Assistant//EN",
    "CALSCALE:GREGORIAN",
    "METHOD:PUBLISH",
    "BEGIN:VEVENT",
    `UID:${escapeIcs(reference)}@multiwoz-restaurant-assistant.local`,
    `DTSTAMP:${formatIcsUtc(new Date())}`,
    `DTSTART;TZID=Europe/London:${formatIcsLocal(start)}`,
    `DTEND;TZID=Europe/London:${formatIcsLocal(end)}`,
    `SUMMARY:${escapeIcs(`Restaurant booking: ${restaurant}`)}`,
    `DESCRIPTION:${escapeIcs(description)}`,
    location ? `LOCATION:${escapeIcs(location)}` : "",
    "END:VEVENT",
    "END:VCALENDAR",
    "",
  ]
    .filter(Boolean)
    .join("\r\n");
}

function getCalendarStart(item) {
  const dateParts = parseBookingDate(item.booking_date) || parseDisplayDate(item.date);
  const timeParts = parseBookingTime(item.time);
  if (!dateParts || !timeParts) {
    return null;
  }
  return { ...dateParts, ...timeParts };
}

function parseBookingDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || ""));
  if (!match) {
    return null;
  }
  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
  };
}

function parseDisplayDate(value) {
  const match = /(?:[A-Za-z]+\s+)?(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})/.exec(String(value || ""));
  if (!match) {
    return null;
  }
  const months = {
    january: 1,
    february: 2,
    march: 3,
    april: 4,
    may: 5,
    june: 6,
    july: 7,
    august: 8,
    september: 9,
    october: 10,
    november: 11,
    december: 12,
  };
  const month = months[match[2].toLowerCase()];
  if (!month) {
    return null;
  }
  return {
    year: Number(match[3]),
    month,
    day: Number(match[1]),
  };
}

function parseBookingTime(value) {
  const match = /^(\d{2}):(\d{2})$/.exec(String(value || ""));
  if (!match) {
    return null;
  }
  return {
    hour: Number(match[1]),
    minute: Number(match[2]),
  };
}

function addMinutes(parts, minutes) {
  const date = new Date(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute + minutes);
  return {
    year: date.getFullYear(),
    month: date.getMonth() + 1,
    day: date.getDate(),
    hour: date.getHours(),
    minute: date.getMinutes(),
  };
}

function formatIcsLocal(parts) {
  return `${parts.year}${pad2(parts.month)}${pad2(parts.day)}T${pad2(parts.hour)}${pad2(parts.minute)}00`;
}

function formatIcsUtc(date) {
  return `${date.getUTCFullYear()}${pad2(date.getUTCMonth() + 1)}${pad2(date.getUTCDate())}T${pad2(
    date.getUTCHours()
  )}${pad2(date.getUTCMinutes())}${pad2(date.getUTCSeconds())}Z`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function escapeIcs(value) {
  return String(value || "")
    .replace(/\\/g, "\\\\")
    .replace(/\n/g, "\\n")
    .replace(/;/g, "\\;")
    .replace(/,/g, "\\,");
}

function safeFilename(value) {
  return String(value || "booking").replace(/[^a-z0-9_-]+/gi, "_");
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showBookingModal(item) {
  if (!bookingModal) {
    return;
  }
  selectedBooking = item;
  const restaurant = item.restaurant_name || item.restaurant || "Restaurant booking";
  bookingModalTitle.textContent = restaurant;
  bookingModalDetails.innerHTML = "";

  const rows = [
    ["Reference", item.reference],
    ["Date", formatDate(item.booking_date, item.day) || item.date],
    ["Time", item.time],
    ["People", item.people ? `${item.people} people` : ""],
    ["Status", item.status],
    ["Cuisine", item.food],
    ["Price", item.pricerange],
    ["Area", item.area],
    ["Address", [item.address, item.postcode].filter(Boolean).join(", ")],
    ["Phone", item.phone],
  ].filter(([, value]) => value);

  rows.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    bookingModalDetails.appendChild(term);
    bookingModalDetails.appendChild(detail);
  });

  bookingModal.hidden = false;
  bookingModalClose?.focus();
}

function hideBookingModal() {
  if (!bookingModal) {
    return;
  }
  selectedBooking = null;
  bookingModal.hidden = true;
}

function applySessionPayload(data) {
  currentSessionId = data.session_id || currentSessionId;
  currentSessionStatus = data.session_status || "active";
  sessionId.textContent = currentSessionId;
  renderHistory(data.messages || [], data.greeting);
  renderBookings(data.bookings || []);
  renderAccountHistory(data.history || {});
  setComposerState();
}

function setComposerState() {
  const isClosed = currentSessionStatus === "closed";
  messageInput.disabled = isClosed;
  sendButton.disabled = isClosed;
  messageInput.placeholder = isClosed
    ? "This conversation has been closed"
    : "Ask for restaurants or bookings";
}

function renderHistory(turns, greeting) {
  messages.innerHTML = "";
  if (!turns.length && greeting) {
    appendMessage("assistant", greeting);
    return;
  }
  turns.forEach((turn) => {
    appendMessage("user", turn.user_message);
    appendMessage("assistant", turn.assistant_message);
  });
}

function renderAccountHistory(history) {
  if (!historyList) {
    return;
  }
  const sessions = history?.sessions || [];
  historyCount.textContent = String(sessions.length);

  historyList.innerHTML = "";
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.textContent = "No conversations yet.";
    historyList.appendChild(empty);
  } else {
    sessions.slice(0, 8).forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `sidebar-history-item history-thread${item.is_current ? " active" : ""}`;
      row.title = "Open this conversation";
      row.addEventListener("click", () => {
        openConversation(item.session_id);
      });
      const title = document.createElement("strong");
      title.textContent = item.last_user_message || "New conversation";
      const meta = document.createElement("span");
      meta.textContent = `${item.turn_count || 0} turns - ${item.booking_count || 0} bookings - ${item.status || "active"}`;
      row.appendChild(title);
      row.appendChild(meta);
      historyList.appendChild(row);
    });
  }
}

function formatDate(isoDate, day) {
  if (!isoDate) {
    return day || "selected day";
  }
  const parts = isoDate.split("-").map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) {
    return isoDate;
  }
  const date = new Date(parts[0], parts[1] - 1, parts[2]);
  return date.toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

function renderBookings(items) {
  bookings.innerHTML = "";
  bookingCount.textContent = String(items.length);

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No bookings yet.";
    bookings.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "booking-card";

    const head = document.createElement("div");
    head.className = "booking-head";

    const titleBlock = document.createElement("div");
    const title = document.createElement("div");
    title.className = "booking-title";
    title.textContent = item.restaurant_name || "Selected restaurant";

    const ref = document.createElement("div");
    ref.className = "booking-ref";
    const refText = document.createElement("span");
    refText.textContent = item.reference;
    ref.appendChild(refText);
    ref.appendChild(makeCopyButton(item.reference));
    ref.appendChild(makeDetailsButton(item));
    ref.appendChild(makeCalendarButton(item));

    titleBlock.appendChild(title);
    titleBlock.appendChild(ref);

    const status = document.createElement("span");
    status.className = `status ${item.status || ""}`;
    status.textContent = item.status || "recorded";

    head.appendChild(titleBlock);
    head.appendChild(status);

    const meta = document.createElement("div");
    meta.className = "booking-meta";
    meta.textContent = `${formatDate(item.booking_date, item.day)} at ${item.time || "time"} for ${
      item.people || "?"
    } people`;

    const detail = document.createElement("div");
    detail.className = "booking-detail";
    const descriptors = [item.pricerange, item.food, item.area].filter(Boolean).join(", ");
    const address = [item.address, item.postcode].filter(Boolean).join(", ");
    detail.textContent = [descriptors, address].filter(Boolean).join(" - ");

    card.appendChild(head);
    card.appendChild(meta);
    if (detail.textContent) {
      card.appendChild(detail);
    }
    bookings.appendChild(card);
  });
}

async function loadSession() {
  const response = await fetch("/api/session");
  if (response.status === 401) {
    window.location.href = "/login";
    return;
  }
  const data = await response.json();
  applySessionPayload(data);
}

async function openConversation(conversationId) {
  if (!conversationId) {
    return;
  }
  const response = await fetch(`/api/session/${encodeURIComponent(conversationId)}`, { method: "POST" });
  if (response.status === 401) {
    window.location.href = "/login";
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    appendMessage("assistant", data.error || "I could not open that conversation.");
    return;
  }
  applySessionPayload(data);
  messageInput.focus();
}

async function sendMessage(message) {
  if (currentSessionStatus === "closed") {
    return;
  }
  appendMessage("user", message);
  sendButton.disabled = true;
  messageInput.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: currentSessionId }),
    });
    const data = await response.json();
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!response.ok) {
      appendMessage("assistant", data.error || "I could not process that message.");
      return;
    }
    currentSessionId = data.session_id || currentSessionId;
    currentSessionStatus = data.session_status || "active";
    sessionId.textContent = currentSessionId;
    appendMessage("assistant", data.response);
    renderBookings(data.bookings || []);
    renderAccountHistory(data.history || {});
  } finally {
    setComposerState();
    if (currentSessionStatus !== "closed") {
      messageInput.focus();
    }
  }
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }
  messageInput.value = "";
  sendMessage(message);
});

newSessionButton.addEventListener("click", async () => {
  const response = await fetch("/api/new-session", { method: "POST" });
  if (response.status === 401) {
    window.location.href = "/login";
    return;
  }
  const data = await response.json();
  applySessionPayload(data);
  messageInput.focus();
});

copyHistoryButton.addEventListener("click", async () => {
  copyHistoryButton.disabled = true;
  const originalText = copyHistoryButton.textContent;
  try {
    const response = await fetch(`/api/session/${encodeURIComponent(currentSessionId)}/export`);
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }
    const data = await response.json();
    if (!response.ok || !data.transcript) {
      copyHistoryButton.textContent = "Copy failed";
      return;
    }
    await copyText(data.transcript);
    copyHistoryButton.textContent = "Copied";
  } catch {
    copyHistoryButton.textContent = "Copy failed";
  } finally {
    window.setTimeout(() => {
      copyHistoryButton.textContent = originalText;
      copyHistoryButton.disabled = false;
    }, 1400);
  }
});

bookingModalClose?.addEventListener("click", hideBookingModal);

bookingModal?.addEventListener("click", (event) => {
  if (event.target === bookingModal) {
    hideBookingModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && bookingModal && !bookingModal.hidden) {
    hideBookingModal();
  }
});

bookingModalCopy?.addEventListener("click", async () => {
  if (!selectedBooking?.reference) {
    return;
  }
  const original = bookingModalCopy.textContent;
  bookingModalCopy.disabled = true;
  try {
    await copyText(selectedBooking.reference);
    bookingModalCopy.textContent = "Copied";
  } catch {
    bookingModalCopy.textContent = "Copy failed";
  } finally {
    window.setTimeout(() => {
      bookingModalCopy.textContent = original;
      bookingModalCopy.disabled = false;
    }, 1200);
  }
});

bookingModalCalendar?.addEventListener("click", () => {
  if (!selectedBooking) {
    return;
  }
  const event = buildCalendarEvent(selectedBooking);
  if (!event) {
    bookingModalCalendar.textContent = "Missing date";
    window.setTimeout(() => {
      bookingModalCalendar.textContent = "Download calendar";
    }, 1200);
    return;
  }
  const filename = `${safeFilename(selectedBooking.reference || "booking")}.ics`;
  downloadTextFile(filename, event, "text/calendar;charset=utf-8");
});

loadSession().catch(() => {
  appendMessage("assistant", "The web app could not load the current session.");
});
