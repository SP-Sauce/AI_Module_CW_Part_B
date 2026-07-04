# Part A Summary

## Coursework Direction

Part A selected the Large Language Model based dialogue system topic for the
6CM606 Frontiers in Artificial Intelligence and Data Science assessment. The
original design explored a broader travel-package assistant using MultiWOZ
domains such as hotel, restaurant, train, attraction and taxi.

For Part B, the prototype is deliberately narrowed to a restaurant search and
simulated booking assistant. This keeps the build feasible, demonstrable and
aligned with the strongest EDA finding for the selected task.

## EDA Findings Used

The Part A EDA compared MultiWOZ with the Bitext customer support dataset.
Bitext had 26,872 single-turn support records and was clean, but it was less
suitable for multi-turn dialogue state tracking.

MultiWOZ was selected because it contains multi-turn, task-oriented
conversations. The flattened EDA found:

| Service | Dialogue count | Service-turn count |
| --- | ---: | ---: |
| restaurant | 4,728 | 68,234 |
| hotel | 4,182 | 66,816 |
| train | 3,931 | 60,354 |
| attraction | 3,485 | 53,648 |
| taxi | 1,872 | 30,654 |
| hospital | 108 | 770 |
| bus | 6 | 108 |

Restaurant is the largest service by both dialogue involvement and service-turn
count. This justifies focusing the implementation on restaurant search rather
than spreading a small prototype across hotel, train, taxi, bus or hospital.

The EDA also found that MultiWOZ utterances are short and task-focused. Overall
mean utterance length was 13.51 words and the restaurant service mean was 13.42
words. Dialogue length averaged 13.71 turns, supporting a system with session
state rather than isolated one-turn responses.

## Design Consequences

- Use MultiWOZ restaurant records as the grounding source.
- Track restaurant preferences across turns: food, area, price range, day, time
  and number of people.
- Use rule-based slot extraction as a robust baseline and optional LLM support
  for generation.
- Use transparent retrieval and ranking so failures are explainable in the demo.
- Keep bookings simulated and session-only.
- Avoid claims about live availability, payments, external reviews or verified
  dietary certification.

## Assessment Alignment

The assessment brief asks the LLM dialogue-system component to interpret
customer queries, generate clear responses and maintain coherent conversation
flow. This project addresses those requirements through:

- intent and slot extraction;
- dialogue state tracking;
- TF-IDF restaurant retrieval;
- preference-fit ranking;
- grounded response generation;
- simulated booking, rescheduling and cancellation;
- evaluation covering slots, retrieval, task success and latency.

