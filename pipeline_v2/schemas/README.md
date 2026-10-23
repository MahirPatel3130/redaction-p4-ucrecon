# V2 data contracts

The authoritative schemas are strict Pydantic models in
`src/academic_pipeline_v2/models.py`. Model-facing constrained-generation schemas are defined in
`semantic.py` and `privacy.py` and are sent directly to the local vLLM server using
`response_format=json_schema`.

Exported schemas can be regenerated in an installed development environment with:

```bash
python scripts/export_schemas.py --output schemas/generated
```
