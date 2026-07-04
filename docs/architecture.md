# Architecture

## Pipeline

```text
User message
-> intent and slot extraction
-> dialogue state update
-> restaurant retrieval
-> preference-fit ranking
-> grounded response generation
-> simulated booking update
```

The assistant is a CPU-feasible retrieval-grounded dialogue system. It does not
train an LLM from scratch and does not require model downloads for the default
test path.

## Components

`slot_extraction.py` detects intents and extracts restaurant slots using
validated rules. It supports food type, area, price range, day, time and number
of people. Optional LLM extraction can be added later, but rule extraction is the
tested baseline.

`dialogue_state.py` stores session-level context. The state object has food,
area, price range, booking day, booking time, people, selected restaurant,
booking status, simulated reference and conversation history.

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

`booking.py` creates simulated booking references such as `SIM-AB12CD`, supports
simulated rescheduling and cancellation, and avoids any claim that a real booking
was made.

`assistant.py` coordinates the pipeline and returns optional debug information
for live demonstration.

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
availability, process payments, use external review sites or store personal data
beyond the current Python session.

