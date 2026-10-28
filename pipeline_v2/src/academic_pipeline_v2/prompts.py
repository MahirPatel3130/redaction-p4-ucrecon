SYSTEM = """You extract source-grounded data for an offline protected research pipeline.
Return only JSON matching the supplied schema. Never infer a value not present in SOURCE_TEXT.
Every found value and every identifier mention must include exact document-global character offsets.
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

PRIVACY_PROMPT = """TASK=privacy_detection_only
PASS={pass_name}
CHUNK_ID={chunk_id}
POLICY_REPLACE_TYPES={replace_types}
Inspect the numbered lines of candidate de-identified content. This is detection only: never
rewrite the content and never calculate character offsets. For each visible violation, copy the
exact visible text and its numbered source line. Use finding_kind=contextual_risk only when a
credible identifying combination has no single visible span.
Opaque placeholders like [[PERSON:person_...]], [[ORG:org_...]], and [[WORK:work_...]] are safe.
Mark safe=true only when no visible violation or contextual risk remains.
Return at most 50 unique findings in document order.
Use only these identifier_type values: person_name, organization, work_title, email,
phone_number, postal_address, website, unique_identifier, signature, date. Never report any span
inside an opaque [[...]] placeholder. Do not report inferred names, paraphrases, pronouns, common
nouns, generic roles, generic academic terms, ordinary evaluation prose, or non-identifying dates.
Only categories listed in POLICY_REPLACE_TYPES are prohibited visible identifiers. Use
finding_kind=contextual_risk for a credible linkable combination that the policy does not express
as one prohibited visible span.

CANDIDATE_TEXT:
{text}
"""
