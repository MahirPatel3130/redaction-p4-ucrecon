# Pipeline V2 architecture

## Data flow

```text
PDF inventory
  -> canonical native/OCR text (protected, cached)
  -> constrained text-model extraction (protected, cached)
  -> SQLite entity resolution + deterministic substitution
  -> deterministic and model privacy review
  -> researcher candidate
  -> sampled approval
  -> explicit researcher release
```

The source PDF is opened only by canonical extraction. Semantic extraction, identity resolution,
privacy repair, policy changes, and exports reuse canonical text. A PDF is revisited only when its
checksum changes or the extraction/OCR stage fingerprint changes.

## Model boundary

The text model identifies semantic records and exact source spans. It never rewrites letter prose,
assigns final IDs, substitutes text, updates registries, or decides package integrity. Those tasks
are deterministic Python operations. Native extraction is preferred; PaddleOCR is invoked only for
pages that fail text-quality checks.

The production adapter accepts loopback vLLM endpoints only. One resident server supplies
continuous batching while the pipeline submits length-bounded requests concurrently. The mock
backend supplies deterministic responses for network-free tests.

## State and invalidation

Each document has separate canonical, semantic, and privacy artifacts. Every artifact has its own
fingerprint and checksum. Model changes invalidate semantic/privacy work but retain canonical text;
privacy-policy changes retain canonical and semantic work; export changes invoke no model.

`state/registry.sqlite` is the protected authority for entities, aliases, mentions, relationships,
and optional v1 imports. Research outputs contain only opaque entity IDs.

## Privacy boundary

All canonical text, original semantic values, aliases, mappings, redactions, checkpoints, and QA
findings remain under `protected/` with owner-only permissions. Candidate exports are scanned for
direct identifiers and every protected alias before a review sample is created. Final export is a
separate command bound to the exact candidate-manifest checksum reviewed.

Withheld documents are repaired automatically at the smallest failed stage up to the configured
limit. Remaining unsafe documents are excluded and represented only by non-sensitive quality
metadata; they do not create a manual adjudication queue.
