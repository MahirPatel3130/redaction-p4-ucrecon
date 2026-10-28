# Academic Document Pipeline V2

V2 is an isolated, offline, text-first pipeline for converting P4-sensitive academic CVs and
recommendation letters into a semantic-preserving de-identified research dataset. It does not
import or change the repository's v1 pipeline.

The source PDF is processed once into protected canonical text. Native PyMuPDF extraction is used
when reliable; PaddleOCR handles deficient pages. All later model extraction, identity resolution,
de-identification, finite privacy validation, and export reuse the cached canonical representation.

Privacy validation is evidence-grounded and terminating. Qwen performs two detection-only passes
around at most one deterministic replacement batch. Model offsets are never trusted, unsupported
model claims cannot delete text, and repeated generative repair is prohibited. The terminal states
are `validated_candidate`, `review_required`, and `withheld`. Only `validated_candidate` documents
enter the candidate researcher package; they still require the configured stratified human review
before final export.

## Output boundary

```text
RUN_ROOT/
├── protected/                 # P4: canonical text, values, mappings, checkpoints, QA
├── metrics/                   # sanitized timings and counts
├── researcher_candidate/      # de-identified, pending sampled approval
└── researcher/                # de-identified approved release

STATE_ROOT/
└── registry.sqlite            # P4: stable identities and protected aliases
```

Never transfer `protected/` or `registry.sqlite` as part of the standard researcher delivery.

## Environment

Create the pipeline environment:

```bash
cd pipeline_v2
conda env create -f environments/pipeline.yml
conda activate academic-pipeline-v2
python -m pip install --no-deps -e .
```

On an air-gapped production host, stage all wheels, PaddleOCR models, and LLM weights through the
approved process first. Runtime commands must not download dependencies or models.

Set `ocr.text_detection_model_dir` and `ocr.text_recognition_model_dir` to existing local model
directories. V2 refuses to initialize PaddleOCR without explicit local paths, preventing accidental
runtime downloads. The orientation model path is optional.

Create the separate model-server environment after validating a Blackwell-compatible vLLM/Torch
combination:

```bash
conda env create -f environments/model_server.yml
conda activate academic-pipeline-v2-vllm
```

Start Qwen3-14B from a local model directory. Bind only to loopback:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
vllm serve /protected/models/Qwen3-14B \
  --served-model-name Qwen/Qwen3-14B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768
```

Use the exact vLLM options supported by the version validated on the restricted host; the pipeline
requires only the local OpenAI-compatible chat-completions endpoint and JSON-schema responses.

## Configuration

Copy `configs/default.yaml` to a protected location and set:

- Every recruitment `input_root`.
- A new immutable `output_root` for the run.
- A persistent protected `state_root`.
- The local model name served by vLLM.
- The loopback endpoint and privacy policy path.

Paths are configuration-relative when not absolute. Do not commit production configuration if it
contains protected paths.

## Normal workflow

```bash
academic-pipeline-v2 inventory --config /protected/v2/config.yaml
academic-pipeline-v2 extract --config /protected/v2/config.yaml
academic-pipeline-v2 process --config /protected/v2/config.yaml
academic-pipeline-v2 status --config /protected/v2/config.yaml
academic-pipeline-v2 export --config /protected/v2/config.yaml --candidate
academic-pipeline-v2 validate-release --config /protected/v2/config.yaml
```

`process` also runs canonical extraction when it has not already completed. Canonical extraction,
semantic extraction, and privacy validation are independently cached.

Review only the stratified sample of automatically releasable documents listed in:

```text
RUN_ROOT/protected/review/sample.csv
```

Enter `pass` for every genuinely inspected passing row. A failed sampled document must be repaired;
do not mark it passed. Bind the review to the candidate and create the final package:

```bash
academic-pipeline-v2 finalize-review \
  --config /protected/v2/config.yaml \
  --decisions /protected/v2/runs/run_001/protected/review/sample.csv

academic-pipeline-v2 export --config /protected/v2/config.yaml --release
```

The deliverable is `RUN_ROOT/researcher/`.

## Resume and automatic recovery

After interruption, or to retry only failed/withheld stages:

```bash
academic-pipeline-v2 resume --config /protected/v2/config.yaml
```

Validated and review-required compatible checkpoints are reused. Failed canonical/semantic stages
and withheld privacy stages are retried. `privacy.maximum_repairs` remains configuration-compatible,
but privacy validation performs exactly one deterministic repair batch when it is greater than zero.
There is no model-driven repair loop. Unresolved documents remain withheld without blocking safe
documents.

Example detached production run:

```bash
tmux new-session -d -s academic-v2-process bash -lc '
source /home/USER/miniconda3/etc/profile.d/conda.sh
conda activate academic-pipeline-v2
cd /path/to/repository/pipeline_v2
academic-pipeline-v2 process --config /protected/v2/config.yaml
'
```

Monitor only the sanitized log:

```bash
tail -f /protected/v2/runs/run_001/metrics/pipeline.log
```

## Synthetic tests and smoke run

Tests do not require a GPU, network, or private data:

```bash
python -m pip install -e '.[test]'
pytest
```

Generate fictional PDFs:

```bash
python scripts/generate_synthetic_fixtures.py --output-root fixtures/generated
```

For a complete mock smoke run, copy `configs/benchmark.yaml`, change `model.backend` to `mock`, and
run inventory, process, candidate export, and validation. The mock backend is never selected by a
production configuration unless explicitly requested.

## Benchmarking

Run the same fixture/configuration with the Gemma v1 baseline, Qwen3-14B, and Qwen3-32B. For each
v2 run:

```bash
academic-pipeline-v2 process --config /protected/v2/benchmark_qwen14b.yaml
academic-pipeline-v2 export --config /protected/v2/benchmark_qwen14b.yaml --candidate
academic-pipeline-v2 evaluate --config /protected/v2/benchmark_qwen14b.yaml
```

Compare `metrics/evaluation.json`, `metrics/run_summary.json`, and document timings. Do not select a
winner without paired accuracy, privacy, latency, token, and VRAM measurements.

## Optional v1 identity migration

Preview first:

```bash
academic-pipeline-v2 migrate-v1-registry \
  --config /protected/v2/config.yaml \
  --v1-registry /protected/relationship_state/identity_registry.json \
  --dry-run
```

Apply once after reviewing the summary:

```bash
academic-pipeline-v2 migrate-v1-registry \
  --config /protected/v2/config.yaml \
  --v1-registry /protected/relationship_state/identity_registry.json \
  --apply
```

Normal v2 runs never read v1 state.

## Researcher package

The clean v2 package includes canonical JSONL records for documents, CV profiles, and letters;
normalized CSVs for education, employment, publications, authors, entities, relationships, and
quality; plus documentation and checksums. Original values, paths, filenames, evidence spans,
geometry, mappings, and model responses remain protected.

See [architecture](docs/architecture.md) and [benchmark design](docs/benchmark.md).
