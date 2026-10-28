# Model and performance benchmark

Run identical fixtures for the current Gemma baseline, Qwen3-14B, and Qwen3-32B. Keep OCR output,
privacy policy, prompts, schemas, NER backend, NER threshold, and evaluation ground truth fixed. Test
Qwen3-14B first, then 32B.

Separately compare spaCy `en_core_web_lg` and GLiNER `gliner_medium-v2.1` while holding the Qwen
classifier and every other setting fixed. Use different immutable output roots. The NER comparison
must report candidate recall, false-positive replacements, preserve/review decisions, privacy
withholding, CPU latency, and memory. Do not combine the model-family and NER experiments into one
comparison because their effects would be confounded.

Report field and publication-entry precision/recall, missingness accuracy, span grounding,
relationship accuracy, sensitive-entity recall, residual leakage, false replacement, withholding,
JSON validity, retries, tokens, calls/document, median and p95 latency, throughput, and peak VRAM.

Do not select a production model or claim speedup until paired restricted-host measurements exist.
Synthetic/public accuracy demonstrates implementation behavior, not P4 safety.
