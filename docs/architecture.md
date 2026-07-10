# Architecture

## Pipeline

```text
User message
-> intent and slot extraction
-> dialogue state update
-> restaurant retrieval
-> preference-fit ranking
-> ResponsePlan construction
-> deterministic natural-language generation and safety guard
-> booking-state update
-> SQLite session and booking persistence in the web app
```

The assistant is a CPU-feasible retrieval-grounded dialogue system. It does not
train an LLM from scratch and does not require model downloads for the default
test path.

## Components

`slot_extraction.py` detects intents and extracts restaurant slots. In default
CPU mode it uses validated rules. In LLM mode it uses a Transformers model to
produce compact JSON intent/slot output, then validates and merges that output
with the same rule safeguards. It supports food type, area, price range, day,
time and number of people.

The strict LoRA training path uses the same inference prompt as runtime:

```text
Task: Extract the restaurant assistant intent and slots.
Return only one valid minified JSON object.
Do not explain.
Do not use markdown.
Allowed intents: search, list, restaurant_info, book, update_booking, cancel_booking, booking_list, greeting, thanks, goodbye, unsupported.
Allowed slots: food, food_candidates, cuisine_group, area, pricerange, day, time, people, restaurant_name, booking_reference.
User: <USER_MESSAGE>
JSON:
```

Strict model labels such as `update_booking` and `cancel_booking` are mapped
back to the assistant's internal `reschedule` and `cancel` labels after raw LLM
metrics are recorded.

`dialogue_state.py` stores session-level context. The state object has food,
area, price range, booking day, concrete booking date, booking time, people,
selected restaurant, booking status, booking reference and conversation history.

`retrieval.py` builds searchable text from restaurant records and uses
scikit-learn TF-IDF similarity. Known constraints are used for exact filtering
where possible, with TF-IDF as fallback and tie-breaker.

`ranking.py` gives a transparent score based on exact food, area and price
matches plus text similarity. It returns matched constraints, unmatched
constraints and a short explanation.

`response_plan.py` defines the structured response boundary after state updates
and retrieval. Plans carry the dialogue act, user intent, constraints, public
restaurant evidence, selected restaurant, missing constraints, alternatives,
next action and debug-only warnings/internal notes.

`nlg.py` converts response plans into customer-facing text with deterministic
phrasing. It rejects JSON-like text, raw database/debug fields and ungrounded
optional generation before a message is returned to the user.

`llm_generator.py` produces optional guarded responses. `--enable-llm` is only
for slot extraction; response generation requires `--enable-response-llm`.
The default response model is `google/flan-t5-base`, and a trained response LoRA
adapter can be supplied with `--response-model-name`. Safe deterministic NLG is
always the final fallback, and optional LLM output is discarded when it fails the
response safety checks.

`scripts/build_slot_training_data.py` creates strict train/dev JSONL files for
the slot extractor without copying exact evaluation fixture text. Targets are
canonical minified JSON strings. `scripts/train_strict_slot_extractor_lora.py`
is the Kaggle-friendly LoRA training path for `google/flan-t5-base`; it evaluates
raw generated JSON on the dev set before repair/fallback. `scripts/evaluate.py`
therefore reports raw strict model metrics separately from final repaired and
fallback metrics.

`scripts/train_qlora_slot_extractor.py` is an optional PEFT/QLoRA training path
for the JSON slot-extraction model. It saves adapters under `models/`, which can
then be supplied with `--slot-model-name`. FLAN-T5-small is the lightweight
baseline; the Colab workflow can additionally train the stronger
FLAN-T5-base adapter under `models/slot-extractor-qlora-base`. Both use the same
validation, constrained JSON repair, weak-repair detection and rule fallback.
The base-model experiment trains on deterministic template augmentation from
`scripts/augment_slot_training_data.py`; the generator validates compact JSON
targets and rejects normalized overlap with the separate hold-out fixture.

`booking.py` creates booking references such as `BK-AB12CD`, supports
rescheduling and cancellation, and avoids any claim about live restaurant
availability.

`storage.py` persists local user accounts, web sessions, chat turns and booking
records in SQLite. Passwords are stored as hashes. The database stores generated
session ids and restaurant booking details, but no payments, live availability
or external customer account integrations.

`assistant.py` coordinates the pipeline and returns optional debug information
for live demonstration.

`web_app.py` exposes the browser interface using Flask. Users register or log in
locally, the browser keeps a session id cookie for the active conversation, and
each booking record is associated with that conversation. The account history
view groups conversations and recent bookings by user. The local `/admin`
overseer dashboard reads the same SQLite store and presents all demo session
transcripts, booking records and aggregate metrics. The shared login page routes
the local admin account to the dashboard and normal users to the chat app.

## Data Flow

Raw MultiWOZ restaurant records are loaded from `restaurant_db.json` by
`scripts/prepare_multiwoz.py`. Cleaned records are written to
`data/processed/restaurants.jsonl`. Tests and demos can run from
`data/samples/sample_restaurants.json` without the full dataset.

Normalized fields are added for matching, while display fields are preserved for
responses. This lets the assistant avoid inventing addresses, phone numbers,
postcodes or food types.

## Scope Boundaries

The implementation only supports the restaurant domain. It does not provide
hotel, train, taxi, bus, attraction or hospital support. It does not query live
availability, process payments or use external review sites. The web app stores
only local proof-of-concept accounts, session ids, chat turns and booking
records.
