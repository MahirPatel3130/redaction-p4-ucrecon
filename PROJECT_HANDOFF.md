# Project Handoff: P4 Academic Application De-identification and Research Dataset

Last updated: 2026-11-11 (America/Los_Angeles)  
Repository: `MahirPatel3130/redaction-p4-ucrecon`  
Current branch: `main`  
Current implementation commit before this handoff: `bb0fae9` (`Repair incomplete V2 documents and export validation`)

## 1. Purpose of this document

This is the complete engineering handoff for the project. It is intended to let another authorized
engineer or coding agent continue without reconstructing the project from chat history.

This file intentionally does **not** contain actual P4 values, source-document text, credentials,
private keys, model tokens, protected aliases, or raw model responses. Those values belong only on
the restricted machine. Paths, opaque counts, failure codes, architecture, commands, and design
decisions are included because they are necessary to operate and improve the system.

## 2. Actual project goal

The project began as PDF redaction, but its real goal is broader:

1. Ingest academic job applications organized as one applicant directory containing one CV and
   normally three recommendation letters.
2. Detect and remove direct and linkable identifiers from CVs and recommendation letters.
3. Preserve the useful semantic content of the documents rather than returning blacked-out PDFs
   alone.
4. Parse CV education, employment, publications, research fields, and research methods into a
   structured research dataset.
5. Preserve recommendation-letter prose with identifiers replaced by stable semantic placeholders.
6. Create stable opaque IDs for applications, documents, people, organizations, and works so
   researchers can analyze relationships without seeing the underlying identities.
7. Support relationships within and across applications, especially:
   - `wrote_recommendation_for`
   - `listed_as_reference_by`
   - `educated_at`
   - `worked_at`
   - `authored`
   - `published_in`
   - repeat recommenders who wrote for two or more distinct applicants
8. Retain a protected inventory of every removed value so an explicitly authorized P4 research
   package can be produced later without rerunning the model.
9. Produce a clean, documented, de-identified researcher package that contains no direct
   identifiers, protected paths, filenames, geometry, raw model output, or redaction mappings.
10. Record per-document model timings, token counts, retry counts, failure stages, and aggregate
    throughput so future full-scale UCR economics processing can be estimated.

This is a semantic-preserving de-identification pipeline, not merely a PDF-redaction utility.

## 3. Security and data-handling boundary

- Source PDFs and all extracted source text are P4-sensitive.
- Redaction JSON containing actual detected values is also P4-sensitive.
- Protected identity registries and evidence are P4-sensitive.
- The V2 `protected/` tree and `STATE_ROOT/registry.sqlite` must remain on the restricted machine.
- Never commit real application PDFs, document text, detected values, model responses, registries,
  production configuration files, screenshots containing P4, or run outputs.
- Never log source text or detected values. Logs may contain opaque IDs, counts, timings, stages,
  safe error codes, and status values.
- The normal researcher package is de-identified but still controlled because stable opaque IDs are
  linkable.
- A package containing the removed P4 values may be given only to a team explicitly authorized to
  receive P4 data.
- Source PDFs must never be overwritten.
- Failed safety validation must withhold unsafe content rather than lowering the privacy threshold.

## 4. Repository generations

The repository contains three historical generations. They remain available for reproducibility,
but V2 is the current architecture.

### 4.1 Legacy document-specific redaction scripts

- `redact_cv.py`
  - Gemma 4 Vision page-image extraction and CV redaction.
  - Detects applicant name, email, phone, website, address, coauthors, references, publication
    information, and other identifying data.
  - Grounds model detections against PDF text and geometry.
  - Writes protected JSON and can produce permanently black-redacted PDFs.
  - Supports `--upgrade-json-from` to enrich older CV JSON with stable application, document,
    entity, local-person, and mention IDs without rerunning Gemma.
- `redact_recommendation_letters.py`
  - Processes exactly three non-CV PDFs per application in the original workflow.
  - Detects direct identifiers: person names, email, phone, website, postal address, unique ID, and
    signature.
  - Writes one JSON per letter plus `recommendation_redactions.json`.
  - Uses application-local applicant IDs and document-local recommender IDs.
- `resolve_identities_and_relationships.py`
  - CPU-only postprocessor for the legacy CV and recommendation aggregates.
  - Produces protected identity resolution, a de-identified researcher graph, repeat-recommender
    statistics, and an optional authorized `researcher_p4/` package.
  - Exact verified email/unique ID are strongest evidence; balanced automatic matching also
    supports exact full names, valid phones, and personal/profile URLs with recorded confidence.
  - Addresses, signatures, bare institutional domains, and fuzzy names alone never establish
    identity.

### 4.2 Unified V1 semantic pipeline

- Entry point: `build_research_dataset.py`
- Package: `research_pipeline/`
- Model: `google/gemma-4-31B-it` through Transformers vision inference.
- Added CV semantic parsing, full recommendation-letter text, protected redaction inventory,
  placeholders, relationship data, metrics, and researcher packaging.
- It proved the desired product but was slow because it rendered every page and repeatedly invoked
  Gemma vision. Remote measurements also exposed model JSON failures, costly repair loops, and
  excessive withholding.
- It remains a documented legacy workflow and is not imported by V2.

### 4.3 Current V2 text-first pipeline

- Directory: `pipeline_v2/`
- Installed CLI: `academic-pipeline-v2`
- Python package: `academic_pipeline_v2`
- Python requirement: 3.11+
- Primary LLM backend: a local loopback-only vLLM server.
- Current tested model: Qwen3-14B served as `Qwen/Qwen3-14B`.
- Native text: PyMuPDF.
- OCR fallback: PaddleOCR with explicitly staged local model directories.
- Contracts: strict Pydantic models and generated JSON schemas.
- Identity state: SQLite.
- V2 does not import or modify V1 at runtime.

## 5. V2 architecture

### Stage 1: inventory and canonical text

- Accept multiple recruitment roots.
- Treat each immediate child directory as an application.
- Identify CVs by filename conventions and classify other PDFs as recommendation letters.
- Process unambiguous documents even if an application is incomplete.
- Record missing/multiple CVs, unexpected letter counts, and ambiguous inventory explicitly.
- Assign stable application/document IDs from namespace and relative path, not sensitive content.
- Extract text blocks, page/global offsets, reading order, and geometry with PyMuPDF.
- Preserve reliable native-text pages.
- OCR only deficient or failed pages.
- Distinguish truly blank pages from extraction failures.
- Emit `canonical_page_unrecoverable` when a nonblank page cannot be recovered.
- Cache canonical text by source checksum and extraction fingerprint.
- The source PDF is not needed again after a compatible canonical artifact exists.

### Stage 2: semantic extraction

- Submit canonical text, not rendered page images, to local vLLM.
- Use schema-constrained JSON and Pydantic validation.
- CV output includes explicit education, employment, publication, research-field, and
  research-method sections.
- Every required section has one of `found`, `not_found`, `uncertain`, or `model_failed`.
- Store model confidence exactly as returned; deterministic checks live in separate flags.
- Letter extraction finds grounded entity mentions, roles, aliases, and relationships. The model
  does not rewrite the letter.
- Ground every accepted semantic value to source evidence.
- Bounded correction handles malformed or contract-invalid responses.
- Dense/truncated CVs fall back to section-specific contracts rather than repeating the full CV
  schema. Publications are processed in bounded batches and consolidated deterministically.
- The generation ceiling of 16,384 tokens is not treated as the primary truncation fix.

### Stage 3: resolution and deterministic de-identification

- SQLite stores people, organizations, works, aliases, mentions, relationships, decisions, and
  merge history.
- Preserve an applicant identity structurally across their CV and letters.
- Use document-local IDs when identity is uncertain.
- Replace grounded spans from right to left with stable placeholders.
- Representative placeholders:
  - `[[APPLICANT:person_...]]`
  - `[[RECOMMENDER:person_...]]`
  - `[[PERSON:person_...]]`
  - `[[PERSON:doc_person_...]]`
  - `[[ORG:org_...]]`
  - `[[WORK:work_...]]`
  - `[[EMAIL:person_...]]`
  - `[[PHONE:person_...]]`
  - `[[ADDRESS]]`
  - `[[URL]]`
  - `[[SIGNATURE]]`
- Institution, employer, department, journal, publisher, venue, and work title are opaque in the
  researcher package.
- Education/employment years are bucketed into fixed five-year intervals.
- Publication year remains exact.
- Exact job titles become controlled job categories.
- Degree type, academic field, publication type, authorship order, research fields, and methods
  remain analytically available.

### Stage 4: finite privacy gate and export

The privacy gate is deliberately finite and evidence-based:

1. Use grounded structured fields, deterministic patterns, and protected aliases.
2. Run local offline NER on source text with exact offsets.
3. Directly accept strong prohibited PERSON/ORG/WORK candidates according to policy.
4. Use Qwen only to classify genuinely ambiguous, already-grounded NER candidate IDs.
5. Qwen cannot invent text, offsets, or replacement spans.
6. Union candidates and perform at most one deterministic replacement batch.
7. Verify every replacement and scan residual content for deterministic identifiers and accepted
   protected aliases.
8. End in exactly one status:
   - `validated_candidate`
   - `review_required`
   - `withheld`

The NER model is not repeatedly run over placeholder-modified prose. That previous pattern changed
contextual classifications and created nonterminating/over-redacting behavior.

Application-level contextual re-identification is evaluated separately and is nonblocking. It must
not drive character-span deletion.

Only `validated_candidate` content enters `researcher_candidate/`. Review-required and withheld
documents expose only non-sensitive quality metadata outside the protected boundary.

## 6. V2 typed contracts

Important contracts are in `pipeline_v2/src/academic_pipeline_v2/models.py` and generated under
`pipeline_v2/schemas/generated/`:

- `ApplicationRecord`
- `DocumentRecord`
- `CanonicalDocument` / `CanonicalPage` / `TextBlock`
- `EvidenceSpan`
- `ExtractedField`
- `EntityMention`
- `CVExtraction`
- `LetterExtraction`
- `ResolvedRelationship`
- `PrivacyResult`
- `DocumentCheckpoint`
- `QualityRecord`

Key enum semantics:

- Document types: `cv`, `recommendation_letter`
- Value statuses: `found`, `not_found`, `uncertain`, `not_applicable`, `model_failed`
- Privacy release statuses: `validated_candidate`, `review_required`, `withheld`
- Person roles include applicant, recommender, reference, coauthor, recipient, advisor,
  committee member, third party, institution, and unknown.
- Supported relationships include recommendation, reference, education, employment, authorship,
  and publishing-venue edges.

`model_failed` must have null confidence. A `found` field must have a value. Confidence is supplied
by the model and is not recalculated by code.

## 7. V2 filesystem boundary

```text
RUN_ROOT/
├── protected/                 # P4 canonical text, evidence, values, mappings, QA, checkpoints
├── metrics/                   # sanitized logs, timings, counts, run summary
├── researcher_candidate/      # de-identified candidate pending release review
└── researcher/                # approved de-identified release

STATE_ROOT/
└── registry.sqlite            # P4 stable entity/alias/relationship state
```

Important protected artifacts include:

- `protected/inventory.json`
- canonical-document artifacts
- semantic artifacts
- `protected/checkpoints/<document_id>/checkpoint.json`
- `protected/checkpoints/<document_id>/privacy.json`
- `protected/qa_reports/`
- `protected/contextual_risk/`
- `protected/review/sample.csv`

Important sanitized metrics include:

- `metrics/pipeline.log`
- per-document timing records
- `metrics/run_summary.json`
- evaluation metrics when requested

## 8. Researcher candidate/release contents

The canonical de-identified V2 package is table-oriented and avoids a duplicate all-in-one source
of truth:

```text
researcher_candidate/ or researcher/
├── README.md
├── data_dictionary.json
├── applications.csv
├── documents.jsonl
├── cv_profiles.jsonl
├── recommendation_letters.jsonl
├── education.csv
├── employment.csv
├── publications.csv
├── publication_authors.csv
├── entities.csv
├── relationships.csv
├── quality.csv
└── manifest.json
```

Researcher priorities:

- `cv_profiles.jsonl`: one structured CV profile with explicit section statuses.
- `education.csv`: education histories with de-identified organizations and bucketed periods.
- `employment.csv`: employment histories with organization IDs and controlled categories.
- `publications.csv` and `publication_authors.csv`: works, exact publication years, types, authorship
  order, and opaque person/work/venue IDs.
- `recommendation_letters.jsonl`: source-preserving prose with approved substitutions and page
  markers; unsafe letter text is absent.
- `relationships.csv`: graph edges joining opaque entities.
- `quality.csv`: processing, extraction, privacy, release, repair, and failure status.
- `documents.jsonl`: document-level metadata and safe status.
- `manifest.json`: checksums and package metadata.
- `data_dictionary.json` and generated `README.md`: field definitions and join guidance.

The researcher package must not contain original names, organizations, publication titles, contact
values, protected aliases, source paths, filenames, evidence spans, bounding boxes, local IDs,
model responses, reviewer notes, or replacement maps.

## 9. Candidate review and release policy

- Candidate generation and final release are separate commands.
- Human review applies to a stratified sample of automatically releasable documents, not every
  withheld document.
- The configured default is 10%, minimum 20 documents or all documents if fewer than 20 exist.
- Oversample OCR, repaired, long, and lower-confidence records.
- Enter `pass` only for genuinely inspected passing rows.
- A failed sample triggers targeted correction and resampling of that risk stratum.
- Ambiguous NER findings are never automatically approved.
- If reviewers do not want to resolve ambiguous NER cases, leave those documents
  `review_required`; their unsafe content stays out while safe documents can still be delivered.
- Optional protected document-level NER decisions support only `preserve` or `withhold` and are
  applied with `academic-pipeline-v2 apply-ner-review`.

## 10. Current V2 CLI

```bash
academic-pipeline-v2 inventory --config CONFIG
academic-pipeline-v2 extract --config CONFIG
academic-pipeline-v2 process --config CONFIG
academic-pipeline-v2 resume --config CONFIG
academic-pipeline-v2 status --config CONFIG
academic-pipeline-v2 audit-privacy --config CONFIG --per-type 5
academic-pipeline-v2 apply-ner-review --config CONFIG --decisions DECISIONS.csv
academic-pipeline-v2 export --config CONFIG --candidate
academic-pipeline-v2 validate-release --config CONFIG
academic-pipeline-v2 finalize-review --config CONFIG --decisions REVIEW.csv
academic-pipeline-v2 export --config CONFIG --release
academic-pipeline-v2 evaluate --config CONFIG
academic-pipeline-v2 migrate-v1-registry --config CONFIG --v1-registry V1.json --dry-run
academic-pipeline-v2 migrate-v1-registry --config CONFIG --v1-registry V1.json --apply
```

`resume` reuses compatible successful checkpoints and retries only failed/incomplete work. Model
changes invalidate semantic/privacy artifacts but preserve canonical text. Privacy-policy or NER
changes preserve canonical and semantic work when stage fingerprints are compatible. Export-only
changes require no model calls.

## 11. Local and restricted-machine locations

### Repository locations

- Development workspace used by Codex: `/home/mpate132/redaction_p4_ucrecon`
- Restricted remote clone: `/home/mpate132/redaction-p4-ucrecon`
- Restricted host name observed: `rct.c.ucr-ursa-major-dong-recruit.internal`
- GitHub remote: `MahirPatel3130/redaction-p4-ucrecon`

### Model and environments

- Pipeline environment: `/home/mpate132/miniconda3/envs/academic-pipeline-v2`
- vLLM environment: `/home/mpate132/miniconda3/envs/academic-pipeline-v2-vllm`
- Local Qwen model: `/home/mpate132/models/Qwen3-14B`
- Intended endpoint: `http://127.0.0.1:8000/v1`
- Intended served model name: `Qwen/Qwen3-14B`

### Previously processed sample data

- Sample roots in repository clone: `/home/mpate132/redaction-p4-ucrecon/ucr_production/`
- Recruitment sets used in the 199-document development run:
  - `recruitment_JPF01795`
  - `recruitment_JPF01796`
  - `recruitment_JPF01801`
- V2 sample run root: `/home/mpate132/research_runs_v2/economics_full_001`
- V2 smoke run root: `/home/mpate132/research_runs_v2/economics_smoke_20261023_025408`

### Newly uploaded actual full data

- Uploaded ZIP: `/home/mpate132/protected_incoming/ucr_production_application_files.zip`
- Extracted root: `/home/mpate132/protected_ucr_full_20261110`
- Extraction completion marker: `/home/mpate132/protected_ucr_full_20261110/.extraction_complete`
- Successfully discovered recruitment roots:
  - `/home/mpate132/protected_ucr_full_20261110/ucr_production/recruitment_JPF01795`
  - `/home/mpate132/protected_ucr_full_20261110/ucr_production/recruitment_JPF01796`
  - `/home/mpate132/protected_ucr_full_20261110/ucr_production/recruitment_JPF01801`
  - `/home/mpate132/protected_ucr_full_20261110/ucr_production/recruitment_JPF02019`
- Actual-run configuration: `/home/mpate132/v2_config/economics_actual_full_001.yaml`
- Actual-run output: `/home/mpate132/research_runs_v2/economics_actual_full_001`
- Actual-run state: `/home/mpate132/research_state_v2_actual`
- Pipeline controller log: `/home/mpate132/research_launch_logs/economics_actual_full_001.log`
- vLLM log: `/home/mpate132/vllm_logs/qwen3_14b_actual.log`
- Namespace: `ucr_actual_production_v2_20261110`
- The two de-identified CSV files downloaded alongside the ZIP were intentionally not incorporated
  yet; the current task focuses on application PDFs.

The actual ZIP passed Python `zipfile` integrity testing and was extracted successfully. Do not
upload or extract it again.

## 12. Current blocking infrastructure issue

The actual full run has **not started model inference** because the restricted host currently has
an unusable NVIDIA driver.

Observed diagnostics on 2026-11-10/11:

- Host uptime was about three days.
- Cloud instance: `rct`, zone `us-west3-a`.
- Machine type: `g4-standard-48`.
- PCI hardware is present:
  - `NVIDIA Corporation GB202GL [RTX PRO 6000 Blackwell Server Edition]`
- No `/dev/nvidia*` device files exist.
- NVIDIA kernel modules are not loaded.
- `/proc/driver/nvidia/version` is absent.
- `nvidia-smi` cannot communicate with the driver.
- Installed package state:
  - `nvidia-driver-open 620.70.06-1ubuntu1`
  - dpkg state `iU`, meaning unpacked but not configured.
- Kernel: `7.0.0-1016-gcp`.
- PyTorch in the vLLM environment is `2.13.0+cu130`, but reports CUDA unavailable and zero devices.
- vLLM is `0.32.0`; `vllm._C` was not importable in this failed environment.

Conclusion: the GPU is physically attached, but the kernel driver installation is incomplete or
failed. This cannot be fixed by pipeline configuration. An authorized administrator should finish
the NVIDIA driver installation for the running kernel and reboot. A likely administrative sequence
is `dpkg --configure -a`, dependency repair, verification that the NVIDIA module exists for the
current kernel, and reboot; do not perform that on a shared machine without coordination.

After repair, the first required checks are:

```bash
nvidia-smi
ls -l /dev/nvidia*
/home/mpate132/miniconda3/envs/academic-pipeline-v2-vllm/bin/python -c \
  'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
```

Expected: CUDA true, at least one device, and the Blackwell GPU name. Then verify `vllm._C` and the
server before starting the pipeline. Do not reinstall vLLM blindly until the system driver is fixed.

## 13. Exact continuation after GPU repair

First start vLLM from the validated environment/model path:

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
/home/mpate132/miniconda3/envs/academic-pipeline-v2-vllm/bin/vllm serve \
  /home/mpate132/models/Qwen3-14B \
  --served-model-name Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  >/home/mpate132/vllm_logs/qwen3_14b_actual.log 2>&1 </dev/null &
```

Confirm readiness:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/models
```

Then start or resume the actual run:

```bash
CONFIG=/home/mpate132/v2_config/economics_actual_full_001.yaml
RUN_ROOT=/home/mpate132/research_runs_v2/economics_actual_full_001
LOG=/home/mpate132/research_launch_logs/economics_actual_full_001.log
PIPELINE=/home/mpate132/miniconda3/envs/academic-pipeline-v2/bin/academic-pipeline-v2

if test -f "${RUN_ROOT}/protected/inventory.json"; then
nohup "${PIPELINE}" resume --config="${CONFIG}" >"${LOG}" 2>&1 </dev/null &
else
mkdir -p "${RUN_ROOT}/metrics"
nohup bash -c "set -e; '${PIPELINE}' inventory --config='${CONFIG}'; '${PIPELINE}' extract --config='${CONFIG}'; '${PIPELINE}' process --config='${CONFIG}'" >"${LOG}" 2>&1 </dev/null &
fi
```

Monitor only sanitized logs:

```bash
tail -f /home/mpate132/research_launch_logs/economics_actual_full_001.log
tail -f /home/mpate132/vllm_logs/qwen3_14b_actual.log
```

## 14. Development-run history and measured results

### Gemma V1 measurements

An early 199-document run produced approximately:

- 168 successful documents and 31 failed documents before repairs.
- Mean total time about 396 seconds/document.
- CV mean about 830 seconds/document.
- Recommendation-letter mean about 324 seconds/document.
- About 9 documents/hour and 0.43 pages/minute.
- Initial deterministic/model QA behavior withheld almost every document, which motivated finite,
  grounded privacy validation.

Fast repair improved one V1 run to 118 released and 81 withheld, but the architecture still required
too many repeated model calls and was retained only as legacy evidence.

### V2 two-application smoke development

The V2 smoke set contained two applications, eight documents (two CVs and six letters). During
development it exposed and fixed:

- malformed/truncated structured responses;
- CV output truncation;
- all-null semantic model confidence;
- repeated privacy repair loops;
- model-invented/ungrounded residual findings;
- protected-alias self-triggering;
- over-redaction of organizations, people, work titles, and dates;
- package scans treating opaque join IDs as phone numbers;
- source-document alias scope errors;
- recommendation-letter status consolidation errors;
- export validation failures caused by field-insensitive scanning.

After the grounding/finite-gate fixes, the smoke run reached:

- 8 documents validated;
- 0 review-required;
- 0 withheld;
- 0 failed;
- 31 model calls;
- 8 repairs;
- 111 structured mentions;
- about 28.4 documents/hour on that small mixed workload.

### V2 199-document development run

Before the final targeted repair commit, the representative full development run reported:

- 199 documents total;
- 180 `validated_candidate`;
- 12 `review_required` because of ambiguous NER candidates;
- 7 incomplete/withheld;
- 3 CV semantic failures with `model_output_truncated`;
- 4 recommendation letters with canonical-only artifacts and no privacy artifacts;
- 0 documents withheld by a completed privacy artifact;
- about 46 documents/hour in the reported final retry summary;
- mean about 78 seconds/document, median about 17 seconds/document;
- 566 model calls, 192 repairs, and 1,049 structured mentions.

The 12 ambiguous NER documents remain safe when left `review_required`; they do not have to be
manually approved to deliver the other safe documents.

Commit `bb0fae9` targeted the remaining systemic causes:

1. Recover failed canonical pages with page-level OCR while preserving successful native pages.
2. Never silently skip incomplete canonical documents; use `canonical_page_unrecoverable`.
3. Use section-specific CV fallback extraction and bounded publication batching.
4. Make package validation schema-aware so opaque IDs are format-validated, not scanned as prose.
5. Suppress invalid optional controlled fields to null with explicit quality status rather than
   failing the entire package.
6. Reuse all compatible validated checkpoints and repair only incomplete documents.
7. Support explicit local decisions for ambiguous NER while never auto-approving them.

## 15. Important bugs and why the current design exists

### Page-by-page vision was too expensive

V1 rendered every page and used Gemma as both OCR and semantic parser. V2 extracts native text once,
uses OCR selectively, and sends canonical text to a continuously batched text model.

### Unbounded privacy repair did not converge

Repeated model scans over placeholder-modified prose changed contextual classifications, created
new aliases, and over-redacted content. V2 uses a finite source-grounded NER gate and one
deterministic replacement batch.

### Model findings without source grounding were unsafe and noisy

The model previously returned unsupported person/organization/work findings. V2 requires exact
source evidence and treats unsupported output as advisory or invalid rather than redacting invented
spans.

### Package validation originally scanned IDs as text

Opaque application/document/person IDs could look phone-like. V2 validates known join-key fields by
their ID schema and scans researcher-visible semantic strings and nested free text separately.

### Dense CVs truncated the full schema

Increasing output tokens alone did not solve dense publication lists. V2 uses section-specific
contracts, bounded publication batches, source coverage checks, and deterministic consolidation.

### Confidence was initially null

Prompts/contracts were tightened so the model supplies confidence for found semantic fields.
`model_failed` remains the only expected null-confidence failure status. Code must never invent or
average semantic confidence.

### ZIP extraction operational issue

The restricted machine did not have the `unzip` command. Python's standard `zipfile` module was used
successfully. An incorrect completion marker created after the failed `unzip` attempt was removed,
then recreated only after successful Python extraction.

## 16. Identity and relationship semantics

- `application_id`: stable opaque ID derived from namespace/path structure.
- `document_id`: stable opaque document ID.
- Local person/entity IDs represent parties before cross-document resolution.
- Persistent global person IDs are random opaque IDs in the protected registry, never hashes of raw
  identity values.
- Equality is represented by shared resolved IDs, not an `is_same_person` edge.
- Ambiguous identities remain separate.
- Exact verified email and explicit unique IDs are strongest automatic evidence.
- Exact names, phones, and personal URLs may be used under recorded balanced-match rules in the
  legacy resolver, but V2 registry decisions must remain provenance-backed.
- Addresses and signatures never establish identity.
- Repeat recommender means one resolved recommender has `wrote_recommendation_for` edges to at least
  two distinct applicant IDs; multiple letters for one applicant count once.
- Publications/work records merge by exact DOI, or exact normalized title plus year. Model-proposed
  variants require high confidence and no conflict.
- Organization aliases require exact normalization or conflict-free high-confidence evidence.

## 17. Environment setup

Pipeline environment:

```bash
cd pipeline_v2
conda env create -f environments/pipeline.yml
conda activate academic-pipeline-v2
python -m pip install --no-deps -e .
```

Model-server environment:

```bash
conda env create -f pipeline_v2/environments/model_server.yml
conda activate academic-pipeline-v2-vllm
```

V2 core dependencies are Pydantic 2, PyYAML 6, PyMuPDF 1.26+, and HTTPX. OCR, tests, spaCy NER,
and GLiNER are optional dependency groups. Production OCR and NER paths must point to staged local
weights; runtime downloads are forbidden.

## 18. Testing and verification

Run from `pipeline_v2/`:

```bash
python -m pip install -e '.[test]'
pytest
python -m compileall -q src tests
```

At commit `bb0fae9`, the focused V2 suite passed 38 tests. The final patch added regressions for:

- hidden/unrecoverable page extraction and blank-page distinction;
- section-specific CV publication truncation recovery;
- phone-like opaque IDs in package validation;
- malformed controlled semantic fields;
- checkpoint reuse with zero unnecessary model reruns;
- explicit ambiguous-NER review/withhold behavior.

Before claiming production readiness, still perform a real restricted-host GPU smoke test after the
driver is repaired, inspect the generated candidate package, complete stratified review, validate
the manifest, and scan for leakage.

## 19. Candidate export sequence after processing

```bash
CONFIG=/home/mpate132/v2_config/economics_actual_full_001.yaml

academic-pipeline-v2 status --config="${CONFIG}"
academic-pipeline-v2 export --config="${CONFIG}" --candidate
academic-pipeline-v2 validate-release --config="${CONFIG}"
```

Inspect:

- `RUN_ROOT/researcher_candidate/quality.csv`
- `RUN_ROOT/researcher_candidate/manifest.json`
- `RUN_ROOT/protected/review/sample.csv`
- safe aggregate counts in `RUN_ROOT/metrics/run_summary.json`

After authorized stratified inspection, fill the review decisions, bind them to the candidate, and
export the final package:

```bash
academic-pipeline-v2 finalize-review --config="${CONFIG}" \
  --decisions=/home/mpate132/research_runs_v2/economics_actual_full_001/protected/review/sample.csv

academic-pipeline-v2 export --config="${CONFIG}" --release
```

The standard deliverable is only:

```text
/home/mpate132/research_runs_v2/economics_actual_full_001/researcher/
```

Do not deliver the run root, `protected/`, state registry, logs containing protected diagnostics,
source PDFs, or model-server files.

## 20. What remains intentionally deferred

- Final benchmark-based selection between Qwen3-14B, Qwen3-32B, and the measured Gemma baseline.
- A production accuracy/recall evaluation using authorized reviewed ground truth.
- Incorporating the two separate de-identified application/demographics CSVs into the V2 package.
- Broader fuzzy authorship/advisor/committee identity resolution beyond strong evidence.
- An authorized V2 `researcher_p4/` attachment workflow equivalent to the legacy resolver's P4
  package. V2 already retains the protected evidence needed to build it without reprocessing PDFs.
- Multi-GPU scheduling. V2 uses vLLM continuous batching on one configured GPU; concurrency remains
  configurable.
- Final external delivery policy and storage mechanism.

## 21. Immediate next actions for the next engineer/agent

1. Read this file, `pipeline_v2/README.md`, `pipeline_v2/docs/architecture.md`, and
   `pipeline_v2/configs/privacy_policy.yaml`.
2. Do not touch or re-upload the actual ZIP; extraction is complete.
3. Coordinate NVIDIA driver repair on the `g4-standard-48` VM.
4. Verify `nvidia-smi`, `/dev/nvidia*`, PyTorch CUDA, and `vllm._C` after reboot.
5. Start Qwen3-14B on loopback and verify `/v1/models`.
6. Start/resume `economics_actual_full_001` using its existing config.
7. Monitor sanitized logs and run summary; never print protected artifacts in chat or logs.
8. If failures remain, diagnose by safe failure stage/code and opaque document ID. Do not expose
   document content.
9. Export and validate `researcher_candidate/` only after processing terminates.
10. Resolve systemic errors before sampled release review. Ambiguous NER may remain excluded without
    forcing reviewers to approve it.
11. Complete stratified review and create `researcher/` only after validation passes.
12. Transfer only the approved `researcher/` directory to researchers.

## 22. Git history landmarks

Key milestones, in order:

- `bb261cc`: initial CV redaction prototype
- `c1ee9de`: recommendation-letter workflow
- `2288fbf`: future-ready CV IDs
- `c91a7fe`: direct-person identity resolution
- `58e99c5`: balanced matching and repeat writers
- `e0a73a8`: P4 researcher redaction package
- `571b8ab`: unified semantic research pipeline
- `89c26a5`: isolated V2 foundation
- `69ce8d2`: text-first V2 processing
- `1db7e23`: finite privacy gate
- `5dee4a1`: structured NER gate
- `f6b4e13`: Qwen limited to ambiguous NER candidates
- `fe9125a`: grounded semantic fields and stable final verification
- `104ff75`: required semantic confidence
- `f2a61e3`: bounded semantic schema correction
- `d1b38bc` / `efdcd7a`: recommendation-letter status fixes
- `bb0fae9`: incomplete-document recovery and schema-aware export validation

## 23. Non-negotiable engineering principles

1. Fail closed on privacy uncertainty.
2. Never invent confidence, identifiers, or evidence.
3. Ground every released semantic item to protected provenance.
4. Preserve source prose except approved deterministic substitutions.
5. Keep model judgment bounded; code owns offsets, IDs, replacements, joins, and package integrity.
6. Cache stages independently and never rerun expensive inference without a fingerprint reason.
7. Keep P4 and researcher outputs physically and logically separated.
8. Make every missing section, failed page, withheld record, and unresolved identity explicit.
9. Never treat the absence of a model response as evidence that content is safe.
10. Never weaken release validation merely to increase the released-document count.

