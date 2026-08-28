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
