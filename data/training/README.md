# Slot Extraction Training Data

`slot_instruction_examples.jsonl` is an instruction dataset for the optional
LoRA/QLoRA slot-extraction adapter.

Each row uses:

```json
{"text": "user message", "intent": "search", "slots": {"food": "italian"}}
```

The training script converts each row into the same JSON extraction format used
by the runtime LLM prompt. For a stronger trained adapter, continue extending
this file with labelled MultiWOZ-style restaurant turns before running QLoRA.

The current set includes 230 examples covering:

- incomplete booking requests, such as "book it" or "book pipasha restaurant",
  where the model should return intent `book` with only the supplied slots;
- regional cuisine phrases, such as Middle Eastern, South Asian, East Asian,
  Southeast Asian, North African and West African, where the model should return
  `cuisine_group` plus supported `food_candidates`.
- typo-heavy restaurant turns, short follow-ups, booking changes/cancellations,
  dish-to-cuisine suggestions and restaurant detail questions.

In Colab, install the optional dependencies, upload or clone the repo, then run:

```bash
pip install -r requirements-qlora.txt
python scripts/train_qlora_slot_extractor.py --base-model google/flan-t5-small
```
