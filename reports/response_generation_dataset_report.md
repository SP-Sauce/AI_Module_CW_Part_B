# Response Generation Dataset Report

- Seed: `6062026`
- Source data: `data/processed/restaurants.jsonl`
- Sample data used: `False`
- Source restaurant records: `110`
- Restaurant-level disjoint split achieved: `True`

## Split Counts

| Split | Requested | Actual | Unique Restaurants | Unique Users | Unique Inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 800 | 800 | 38 | 800 | 800 |
| eval | 160 | 160 | 12 | 160 | 160 |
| challenge | 100 | 100 | 10 | 100 | 100 |

## Intent Counts

### train
- `alternative`: 24
- `book`: 144
- `booking_info`: 24
- `booking_list`: 24
- `cancel`: 24
- `cuisine_help`: 24
- `goodbye`: 25
- `greeting`: 25
- `list`: 24
- `reschedule`: 24
- `restaurant_info`: 96
- `search`: 197
- `thanks`: 25
- `unsupported`: 120

### eval
- `alternative`: 5
- `book`: 29
- `booking_info`: 5
- `booking_list`: 5
- `cancel`: 5
- `cuisine_help`: 5
- `goodbye`: 5
- `greeting`: 5
- `list`: 5
- `reschedule`: 5
- `restaurant_info`: 18
- `search`: 40
- `thanks`: 5
- `unsupported`: 23

### challenge
- `alternative`: 3
- `book`: 18
- `booking_info`: 3
- `booking_list`: 3
- `cancel`: 3
- `cuisine_help`: 3
- `goodbye`: 3
- `greeting`: 4
- `list`: 3
- `reschedule`: 3
- `restaurant_info`: 12
- `search`: 24
- `thanks`: 3
- `unsupported`: 15

## Response Category Counts

### train
- `address_info`: 24
- `alternative_suggestions`: 24
- `booking_cancellation`: 24
- `booking_confirmation`: 24
- `booking_info`: 24
- `booking_list`: 24
- `booking_missing_day`: 24
- `booking_missing_multiple`: 24
- `booking_missing_people`: 24
- `booking_missing_time`: 24
- `booking_reschedule`: 24
- `cautious_allergy`: 24
- `cautious_halal`: 24
- `cautious_live_availability`: 24
- `cuisine_help`: 24
- `exact_recommendation`: 25
- `goodbye`: 25
- `greeting`: 25
- `list_results`: 24
- `missing_area`: 24
- `missing_food`: 25
- `missing_multiple_search`: 24
- `missing_price`: 24
- `no_exact_match`: 25
- `no_result`: 25
- `partial_match`: 25
- `phone_postcode_info`: 24
- `thanks`: 25
- `unsupported_hotel`: 24
- `unsupported_payment`: 24
- `unsupported_review`: 24
- `unsupported_taxi`: 24
- `unsupported_train`: 24

### eval
- `address_info`: 5
- `alternative_suggestions`: 5
- `booking_cancellation`: 5
- `booking_confirmation`: 5
- `booking_info`: 5
- `booking_list`: 5
- `booking_missing_day`: 5
- `booking_missing_multiple`: 5
- `booking_missing_people`: 5
- `booking_missing_time`: 5
- `booking_reschedule`: 5
- `cautious_allergy`: 4
- `cautious_halal`: 4
- `cautious_live_availability`: 4
- `cuisine_help`: 5
- `exact_recommendation`: 5
- `goodbye`: 5
- `greeting`: 5
- `list_results`: 5
- `missing_area`: 5
- `missing_food`: 5
- `missing_multiple_search`: 5
- `missing_price`: 5
- `no_exact_match`: 5
- `no_result`: 5
- `partial_match`: 5
- `phone_postcode_info`: 5
- `thanks`: 5
- `unsupported_hotel`: 5
- `unsupported_payment`: 4
- `unsupported_review`: 4
- `unsupported_taxi`: 5
- `unsupported_train`: 5

### challenge
- `address_info`: 3
- `alternative_suggestions`: 3
- `booking_cancellation`: 3
- `booking_confirmation`: 3
- `booking_info`: 3
- `booking_list`: 3
- `booking_missing_day`: 3
- `booking_missing_multiple`: 3
- `booking_missing_people`: 3
- `booking_missing_time`: 3
- `booking_reschedule`: 3
- `cautious_allergy`: 3
- `cautious_halal`: 3
- `cautious_live_availability`: 3
- `cuisine_help`: 3
- `exact_recommendation`: 3
- `goodbye`: 3
- `greeting`: 4
- `list_results`: 3
- `missing_area`: 3
- `missing_food`: 3
- `missing_multiple_search`: 3
- `missing_price`: 3
- `no_exact_match`: 3
- `no_result`: 3
- `partial_match`: 3
- `phone_postcode_info`: 3
- `thanks`: 3
- `unsupported_hotel`: 3
- `unsupported_payment`: 3
- `unsupported_review`: 3
- `unsupported_taxi`: 3
- `unsupported_train`: 3

## Leakage And Safety

- Train/eval input overlap: `0`
- Train/challenge input overlap: `0`
- Eval/challenge input overlap: `0`
- Train/eval user-message overlap: `0`
- Train/challenge user-message overlap: `0`
- Eval/challenge user-message overlap: `0`
- Safety-validation failures: `0`
- Booking-reference grounding failures: `0`

## Limitations

- Responses are synthetic deterministic templates, so they may favour the system's target response style.
- Automatic safety checks do not replace human evaluation of trained model outputs.
