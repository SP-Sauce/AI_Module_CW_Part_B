# Architecture

## Pipeline

```text
User message
-> intent and slot extraction
-> dialogue state update
-> restaurant retrieval
-> preference-fit ranking
-> grounded response generation
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

`dialogue_state.py` stores session-level context. The state object has food,
area, price range, booking day, concrete booking date, booking time, people,
selected restaurant, booking status, booking reference and conversation history.

`retrieval.py` builds searchable text from restaurant records and uses
scikit-learn TF-IDF similarity. Known constraints are used for exact filtering
where possible, with TF-IDF as fallback and tie-breaker.

`ranking.py` gives a transparent score based on exact food, area and price
matches plus text similarity. It returns matched constraints, unmatched
constraints and a short explanation.

`llm_generator.py` produces grounded responses. If Hugging Face Transformers and
a local/downloadable model are available, it can use a small encoder-decoder
model such as `google/flan-t5-small`. Otherwise it uses safe templates. Both
paths are instructed to use only retrieved restaurant evidence.

`scripts/train_qlora_slot_extractor.py` is an optional PEFT/QLoRA training path
for the JSON slot-extraction model. It saves adapters under `models/`, which can
then be supplied with `--slot-model-name`.

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
