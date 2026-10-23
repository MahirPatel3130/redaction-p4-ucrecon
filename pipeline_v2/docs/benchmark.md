# Model and performance benchmark

Run identical fixtures for the current Gemma baseline, Qwen3-14B, and Qwen3-32B. Keep OCR output,
privacy policy, prompts, schemas, and evaluation ground truth fixed. Test Qwen3-14B first, then 32B.

Report field and publication-entry precision/recall, missingness accuracy, span grounding,
relationship accuracy, sensitive-entity recall, residual leakage, false replacement, withholding,
JSON validity, retries, tokens, calls/document, median and p95 latency, throughput, and peak VRAM.

Do not select a production model or claim speedup until paired restricted-host measurements exist.
Synthetic/public accuracy demonstrates implementation behavior, not P4 safety.
