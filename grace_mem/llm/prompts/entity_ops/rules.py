# llm/prompts/entity_ops/rules.py
"""
Entity operation decision rules for ADD vs UPDATE.
"""

ENTITY_OPS_RULES_V2 = """
You will reconcile NEW ENTITIES extracted from the conversation against EXISTING ENTITIES (candidates).
Determine ADD vs UPDATE per item.

Duplicate & Decision Policy:
- Treat as the same entity (UPDATE) only if they clearly refer to the same real-world object or concept — matching name, identity, or key attributes. Type differences alone do NOT imply separation.
- If they are merely related (e.g., different people, versions, or organizations), treat as distinct (ADD).
- SUBJECT GUARD: Events or activities that involve DIFFERENT named persons as their primary subject are ALWAYS distinct entities — choose ADD even if the event type (e.g., "doctor visit", "check-up", "appointment") and phrasing are similar. Person A's check-up and Person B's child's check-up are NOT the same entity.
- Use semantic understanding first; similarity score is only a hint, not proof.
- When uncertain or evidence is weak, choose ADD as a conservative default.

Description Merge Policy:
- merged_description must integrate all key facts from both old and new descriptions — never discard unique details.
- Write a concise, natural English synthesis (1–3 sentences) that combines stable traits from the old record with new contextual info (time, role, version, etc.).
- When overlapping, keep the more specific phrasing; resolve contradictions conservatively, removing only clearly incorrect details.

Canonicalization Policy:
- canonical_name: use the most informative and widely-used form. If the existing candidate has a better canonical name, keep it; otherwise improve it (e.g., expand acronym, add version).
- canonical_type: keep the input type unless an existing entity has a well-established canonical type that better reflects the real-world category (e.g., a "product instance" vs "product line").
- Normalize whitespace; do not include quotes.

Output format:
For each INPUT, output EXACTLY ONE line:
input_name||input_type||action||target_existing_id_or_NULL||canonical_name||canonical_type||merged_description

Hard rules:
- NEVER output the delimiter string '||' inside any field value.
- If action == UPDATE: target_existing_id MUST be EXACTLY one of the IDs in the Valid choices for that INPUT (copy-paste only).
- If action == ADD: target_existing_id MUST be 'NULL'.
- merged_description MUST be non-empty English and is the FINAL consolidated description to OVERWRITE the old one.
- Output ONLY the result lines between the markers:
===BEGIN===
... (lines)
===END===

Quality Guard:
- Ensure merged_description contains at least one fact traceable to the existing description AND at least one fact traceable to the new description (when UPDATE).
- If there are NO candidates, action must be ADD.
"""
