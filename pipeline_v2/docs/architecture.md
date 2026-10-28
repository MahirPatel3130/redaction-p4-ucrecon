# Pipeline V2 architecture

## Data flow

```text
PDF inventory
  -> canonical native/OCR text (protected, cached)
  -> constrained text-model extraction (protected, cached)
  -> SQLite entity resolution + deterministic substitution
  -> structured + offline-NER privacy gate
  -> separate application-level contextual-risk assessment
  -> researcher candidate
  -> sampled approval
  -> explicit researcher release
```

The source PDF is opened only by canonical extraction. Semantic extraction, identity resolution,
privacy validation, policy changes, and exports reuse canonical text. A PDF is revisited only when its
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

The privacy gate has a fixed state machine:

```text
candidate text
  -> grounded semantic fields + deterministic identifiers + protected aliases
  -> local offline NER candidate generation with exact offsets
  -> direct policy handling for strong PERSON/ORG/WORK labels
  -> bounded Qwen classification of ambiguous candidate IDs only
  -> code validation, policy adjudication, deduplication, overlap resolution
  -> zero or one deterministic replacement batch
  -> deterministic residual scan
  -> independent local NER coverage and bounded classification
  -> validated_candidate | review_required | withheld
```

Qwen never discovers spans, supplies offsets, or rewrites prose during privacy validation. It can
only classify source-grounded NER candidate IDs. Missing decisions become review items; confirmed
residuals cause withholding. The model is never called recursively until it returns an empty list.
Without an enabled local NER backend the result is `review_required`, not an automatic pass.

Contextual re-identification is a different question from visible-span removal. V2 records a
nonblocking, application-level cohort assessment under `protected/contextual_risk/`. Small cohorts
are explicitly marked `not_evaluated_insufficient_cohort`; contextual concerns never cause the span
gate to delete additional prose.

Only `validated_candidate` documents enter the researcher candidate package. They remain subject
to the configured stratified release review. `review_required` and `withheld` documents expose only
non-sensitive quality metadata outside the protected boundary.
