# Privacy-gate audit and finite replacement design

## Observed failure

The first V2 privacy gate asked Qwen to discover arbitrary remaining identifiers, replaced its
grounded answers, and then asked the same model again. On the protected smoke run, the returned set
changed across passes. This created hundreds of replacements without a stable stopping condition:
the model was simultaneously proposing evidence, classifying policy, and deciding completeness.

## Current authority order

1. Grounded CV structured fields and grounded letter entity extraction.
2. Deterministic direct-identifier patterns and exact protected aliases.
3. A separately loaded offline NER model that proposes exact source spans. Strong PERSON, ORG, and
   WORK labels covered by policy are accepted directly.
4. Qwen classification of ambiguous supplied candidate IDs only: `replace`, `preserve`, or
   `review`.
5. Deterministic overlap resolution and one right-to-left replacement batch.
6. Final substitution verification plus deterministic identifier and accepted-alias scans.

Qwen cannot invent a privacy candidate, provide an offset, or rewrite the document during this
stage. An invalid NER span is rejected by code. A missing Qwen decision requires review. A confirmed
deterministic or accepted-alias residual is withheld. If NER is unavailable, the document is `review_required`; it does not
silently pass.

## Contextual risk boundary

Contextual re-identification is not represented as a character span. It is assessed separately at
application/cohort level using retained quasi-identifiers and stored under
`protected/contextual_risk/`. It never authorizes deletion of prose. Cohorts below the configured
minimum are explicitly recorded as insufficient for a meaningful uniqueness assessment.

## Audit workflow

Use this protected-only command to sample remaining findings by document and finding type:

```bash
academic-pipeline-v2 audit-privacy --config /protected/v2/config.yaml --per-type 5
```

The resulting CSV includes exact text and context and is P4-sensitive. Reviewers may label entries
as true identifiers, false positives, or ambiguous without changing the researcher export. Those
labels should inform model/threshold benchmarking rather than one-off policy weakening.

## Bounded performance

There are exactly two complementary pre-substitution NER scans: full context and context with known
authoritative spans masked. Their union is classified once in batches controlled by
`ner.candidates_per_call`. Each candidate is tied to an exact validated span. There is at most one
replacement batch, followed only by deterministic verification. Runtime and destructive changes
therefore remain finite regardless of model behavior.
