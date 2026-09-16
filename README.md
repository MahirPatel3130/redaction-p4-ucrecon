# Gemma 4 CV redaction prototype

This offline prototype renders CV PDF pages, asks Gemma 4 Vision for sensitive-region boxes,
snaps those boxes to exact PDF word geometry, adds deterministic email/phone/URL/DOI matches,
writes structured JSON, and permanently removes content in those boxes from a copied PDF.
Every result requires human review; a vision model cannot guarantee P4-compliant recall.

## Setup

Use Python 3.11+ in a fresh environment. Gemma 4 currently requires a recent Transformers build:

```bash
conda create -n cv-redaction python=3.11 -y
conda activate cv-redaction
pip install "transformers>=5.10.1" accelerate torch pillow pymupdf
```

Before the first run, accept Google's Gemma license for `google/gemma-4-31B-it` on Hugging
Face and authenticate with `hf auth login`. Download/cache the model before moving to an
air-gapped sensitive machine. Once cached, use Hugging Face offline mode if required:

```bash
HF_HUB_OFFLINE=1 python redact_cv.py \
  --input-root /protected/applications \
  --output-root /protected/redacted
```

The script defaults to `google/gemma-4-31B-it`, loads it once with automatic device placement,
and accepts `--model` for testing with another compatible Gemma vision checkpoint. Set
`--input-root` to the directory whose immediate child directories are individual applications:

```text
applications/
├── application_001/
│   ├── CV.pdf
│   ├── recommendation_1.pdf
│   └── recommendation_2.pdf
└── application_002/
    ├── curriculum_vitae.pdf
    └── recommendation_1.pdf
```

Each application directory must contain exactly one PDF whose filename contains `CV` or
`curriculum vitae`, case-insensitively. Names such as `CV.pdf`, `Applicant_CV_2025.pdf`, and
`smith-curriculum-vitae.pdf` are accepted. The script processes that CV only; recommendation
letters and unrelated PDFs beside it are ignored. Applications with no matching CV or multiple
matching CVs are logged and skipped rather than guessed. Nested directories below an application
directory are not searched.

For entity-inventory work without PDF generation, use `--json-only`. The resulting JSON includes
PDF-grounded values, canonical entities with aliases and occurrences, per-category coverage,
confidence/evidence metadata, review flags, and rejected ungrounded model detections. Protected
saved responses can be reprocessed without GPU inference using:

```bash
python redact_cv.py --input-root samples --output-root outputs-json \
  --reuse-model-responses-from outputs-v2 --json-only
```

Existing clean CV JSON can be upgraded with future-ready identity fields without source PDFs,
Gemma, or GPU inference. Always use a new output directory so the verified input stays unchanged:

```bash
python redact_cv.py \
  --upgrade-json-from /protected/outputs-v7 \
  --output-root /protected/outputs-v8
```

The enriched record retains the original CV fields and adds `schema_version`, `application_id`,
`document_id`, `document_type`, `parties`, local person IDs, mention IDs, `person_id: null`, and a
reserved empty `relationships` list. `person_id` remains unset until a later protected
cross-application identity-resolution phase. The path-keyed `redactions.json` remains available,
and `cv_redactions.json` additionally groups the same records by application and document.

Outputs preserve the source folder structure. Each CV gets a JSON record and, only when every
page succeeds and post-redaction verification passes, a `_redacted.pdf`. The output root also
receives `redactions.json`, keyed by each source PDF's relative path. JSON separates unique
`entities` from their page-level `occurrences` and retains a flat `redactions` list for geometry
review. Logs intentionally contain counts and paths but not detected text.

Use `--debug-artifacts` only in a protected output directory to retain raw model responses for
malformed-JSON diagnosis. For a detached GPU run:

```bash
tmux new-session -d -s cv-redaction \
  'cd /home/mpate132/redaction_p4_ucrecon && nvidia-modprobe -u -c=0 && \
   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
   /home/mpate132/miniconda3/envs/deceptive_alignment/bin/python redact_cv.py \
   --input-root samples --output-root outputs-v2 --debug-artifacts \
   > redaction-v2.log 2>&1'
```

## Recommendation letters

`redact_recommendation_letters.py` is a separate direct-identifier workflow and does not change
the CV output format. Point `--input-root` at the same applications directory. Each immediate
application folder must contain exactly three non-CV PDFs; folders with another count are recorded
as `invalid_letter_count` and skipped.

JSON is the default output:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python redact_recommendation_letters.py \
  --input-root /protected/applications \
  --output-root /protected/recommendation-results
```

Add `--write-redacted-pdfs` to also create permanently black-redacted PDFs. The script detects
person names, emails, phone/fax numbers, websites, postal addresses, explicit personal IDs, and
signatures. It intentionally preserves standalone institutions, departments, titles, dates,
publications, logos, and evaluation prose.

Each letter receives a JSON file, and `recommendation_redactions.json` groups all documents by
application. Stable application/document IDs, application-local applicant IDs, document-local
recommender IDs, and `person_id: null` fields make the schema ready for a later protected identity
resolution step without attempting cross-application matching now. All output JSON remains P4
sensitive because it contains the detected values. Use `--debug-artifacts` only when protected raw
Gemma responses are needed for diagnosis.

## Direct-person identity resolution

After both JSON inventories are complete, run the CPU-only resolver. It does not load Gemma, read
source PDFs, or modify either redaction output:

```bash
python resolve_identities_and_relationships.py \
  --cv-json /protected/project_run/01_cv_redaction/cv_redactions.json \
  --recommendation-json /protected/project_run/02_letter_redaction/recommendation_redactions.json \
  --registry /protected/relationship_state/identity_registry.json \
  --output-root /protected/project_run/03_identity_resolution/run_001
```

The persistent registry assigns opaque `person_...` IDs. Exact PDF-verified email and unique-ID
matches merge automatically at confidence `1.0`. The balanced automatic tier also links exact
full names, valid phone numbers, and personal/profile URLs at lower recorded confidence. Fuzzy
names require supporting phone or eligible URL evidence. Bare institutional domains, addresses,
fuzzy names alone, ambiguous CV-reference grouping, and hard identifier conflicts never merge.

`restricted/review_queue.csv` retains those exceptional cases as optional advisories; it does not
block delivery. If an authorized reviewer later chooses to adjudicate one, they may set `decision`
to `accept`, `reject`, or `defer`, add an optional `reviewer_note`, and run with a new output
directory and:

```bash
--decisions-csv /protected/project_run/03_identity_resolution/run_001/restricted/review_queue.csv
```

For multiple disjoint batches, repeat both aggregate options. Application folders must be unique
across batches; the resolver rejects duplicate application or document IDs and produces one
combined dataset:

```bash
python resolve_identities_and_relationships.py \
  --cv-json /protected/output_1795_upgraded/cv_redactions.json \
  --cv-json /protected/output_1796_upgraded/cv_redactions.json \
  --cv-json /protected/output_1801_upgraded/cv_redactions.json \
  --recommendation-json /protected/recommendation_1795/recommendation_redactions.json \
  --recommendation-json /protected/recommendation_1796/recommendation_redactions.json \
  --recommendation-json /protected/recommendation_1801/recommendation_redactions.json \
  --registry /protected/relationship_state/identity_registry.json \
  --output-root /protected/relationship_results/run_001
```

The same protected registry may also be reused for a later incremental batch. Strong verified
identifiers in the new batch are compared with active people already stored in the registry, so
previous person IDs remain stable across runs.

The `restricted/` directory and registry remain P4-sensitive because they contain raw identity
evidence. `automatic_matches.csv` audits every automatic merge and its confidence. The
`researcher/` directory contains a de-identified graph in canonical `dataset.json` plus
`applications.csv`, `documents.csv`, `people.csv`, `relationships.csv`, and
`repeat_recommenders.csv`. It supports `wrote_recommendation_for` and
`listed_as_reference_by`. The researcher package is always `ready` with `review_required: false`;
any unresolved cases are reported only as `advisory_count` and remain separate.

People records include identity-linkage confidence, recommendation-letter count, distinct
applicants recommended, and `is_repeat_recommender`. A repeat recommender is a resolved person
connected by recommendation letters to at least two distinct applicant person IDs; multiple
letters for one applicant count once. Stable pseudonymous IDs remain linkable data, so the
researcher package is still a controlled dataset even though it excludes direct identifiers.

Every resolution run must use a new or empty output directory. Keep the registry at a stable,
access-controlled path so person IDs and accepted/rejected decisions survive later runs. The v1
resolver intentionally excludes publications, authorship, advisors, and committee relationships.

Example researcher questions include grouping `relationships.csv` by `subject_person_id` to find
one recommender connected to multiple applicants, or selecting people who occur as both a
`recommender` and `reference` after protected identity resolution.

## Public sample PDFs

The files under `samples/` come from public university career guides containing fictional or
example CV content. Each application folder keeps the unmodified guide as `source_original.pdf`;
its operational `CV.pdf` is a lossless extraction of only the sample-CV pages to avoid wasting model
inference on unrelated guidance. Source URLs and page ranges are recorded in `samples/SOURCES.md`.
They are test data, not P4 data.

## Security notes

- The generated PDF uses PyMuPDF redaction annotations followed by permanent redaction, not blur.
- The JSON intentionally contains detected sensitive strings and must receive the same protections
  as the source data.
- Failed or malformed model output produces a failure record and no partially redacted PDF.
- Remaining extractable email, phone, URL, DOI, or recorded entity text causes verification
  failure and the candidate PDF is withheld.
- Inspect every page and validate sensitive strings are absent before releasing any output.
