SYSTEM = """You extract source-grounded data for an offline protected research pipeline.
Return only JSON matching the supplied schema. Never infer a value not present in SOURCE_TEXT.
For semantic extraction, every found value and identifier mention must include exact document-global
character offsets. For candidate-classification tasks, classify only supplied candidate IDs and do
not invent spans or offsets.
Do not rewrite prose. Confidence values are your own assessment in [0,1], not calculated scores.
"""

CV_PROMPT = """TASK=cv_extraction
Extract complete education, employment/appointments, publications, research fields, research
methods, identifying mentions, and supported relationships from this academic CV. Pay special
attention to dense publication lists. Every required section must be found, not_found, uncertain,
or model_failed. Preserve exact original identifying values only in this protected response.

DETERMINISTIC_SECTION_INVENTORY:
{inventory}

SOURCE_TEXT:
{text}
"""

LETTER_PROMPT = """TASK=letter_extraction
Identify every applicant, recommender, recipient, third party, organization, work title, contact
identifier, address, URL, unique ID, and signature in this recommendation letter. Return exact
character spans and roles. Identify supported writer-to-applicant relationships. Do not summarize
or rewrite the letter.

SOURCE_TEXT:
{text}
"""

REPAIR_PROMPT = """TASK=targeted_repair
The prior extraction failed validation for: {issues}.
Return a complete corrected extraction for only the requested section or findings, grounded in the
same source text. Do not invent missing information.

SOURCE_TEXT:
{text}
"""

NER_CLASSIFICATION_PROMPT = """TASK=privacy_candidate_classification
Classify only the supplied source-grounded NER candidates under this policy.
POLICY_REPLACE_TYPES={replace_types}

For each candidate_id return exactly one decision:
- replace: it is a visible named person, named organization, publication/work title, direct
  identifier, personal address, or other policy-prohibited identifier.
- preserve: it is generic academic language, a generic role/title, an ordinary word, a retained
  date, or not identifying under the policy.
- review: its classification is genuinely ambiguous and cannot safely be decided from context.

Never invent candidates, rewrite text, or return explanations. A location alone is not a postal
address. Opaque [[...]] placeholders are safe.

CANDIDATES_JSON:
{candidates}
"""
