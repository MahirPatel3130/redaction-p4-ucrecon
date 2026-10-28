# Privacy-gate root-cause audit and replacement design

## Observed failure

The original V2 privacy stage performed `detect -> replace -> detect` until the configured repair
limit. On the protected smoke run, semantic extraction completed but all eight documents were
withheld. Repeated passes reported changing sets of names, organizations, dates, and ungrounded
values while adding many replacement spans. Increasing the repair limit reduced semantic retention
without establishing a stable safety condition.

## Root causes in the former implementation

- `validate_privacy` used an LLM returning no findings as the stopping condition.
- Model-selected offsets and exact text jointly authorized deletion, even though offsets were
  unstable and classifications could be false positives.
- Repaired prose was fed back into the same detector, allowing new findings on every iteration.
- Unsupported model claims blocked release but could not be repaired safely.
- Policy decisions and detection were conflated: broad model categories implicitly determined what
  must be removed.
- A 50-finding response could be followed by another destructive pass rather than producing an
  explicit capacity/review state.

The protected smoke evidence was consistent with placeholder/fragment sensitivity, duplicate or
changing model classifications, and hallucinated values. It did not justify weakening deterministic
checks or assuming that a larger model alone would fix the stopping condition.

## Replacement invariants

1. The model proposes candidate evidence; it is not the release authority.
2. Model character offsets are forbidden. Code grounds exact visible text to numbered source lines.
3. Valid placeholders are structurally masked and cannot become findings.
4. Two bounded discovery passes inspect the same candidate. The second masks already discovered
   spans while preserving offsets so the model can find remaining visible candidates.
5. Only grounded, policy-approved spans can be replaced automatically.
6. All accepted spans are deduplicated and replaced in one right-to-left batch.
7. Unsupported model claims are retained as protected advisories and never delete text.
8. Contextual claims require grounded visible evidence; unsupported contextual claims are advisory.
   Grounded contextual risks, capacity saturation, and inconsistent unsafe-without-evidence responses enter
   `review_required`.
9. Confirmed residual direct identifiers or protected aliases enter `withheld`.
10. A clean final deterministic scan plus independent detection-only review produces
   `validated_candidate`; sampled human approval is still required for final export.
11. Candidate and final content hashes, model-call counts, token counts, findings, and repair counts
    remain available in protected metrics without logging P4 values.

## Performance boundary

The number of privacy calls is exactly three per text chunk: discovery, masked coverage, and
independent final review. There is at most one repair batch. Long documents are split into non-overlapping numbered
line chunks; findings are grounded to document-global offsets and deduplicated. This makes runtime
and destructive transformations bounded independently of model behavior.
