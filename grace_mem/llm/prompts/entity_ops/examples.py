# llm/prompts/entity_ops/examples.py
"""
Few-shot examples for entity operation decisions.
Demonstrates correct ADD vs UPDATE behavior with proper merging.
"""

ENTITY_OPS_FEW_SHOT = """FEW-SHOT EXAMPLES

# Case A: UPDATE with real merge (explicitly combine OLD + NEW facts)
[INPUT]
name=PyTorch | type=product | desc=Deep learning framework widely used for computer vision and NLP, known for flexibility and dynamic computation graphs.
Candidates:
- id=product_PyTorch | name=PyTorch | type=product | score=0.87 | desc=An open-source deep learning framework originally developed by Facebook AI Research in 2016, with strong GPU acceleration.
Valid target_existing_id choices: ["product_PyTorch"]

EXPECTED OUTPUT (between markers):
===BEGIN===
PyTorch||product||UPDATE||product_PyTorch||PyTorch||product||An open-source deep learning framework originally developed by Facebook AI Research in 2016, known for flexibility, dynamic computation graphs, strong GPU acceleration, and wide use in computer vision and NLP.
===END===

# Case B: ADD (candidates exist but refer to DIFFERENT entities)
[INPUT]
name=Gemini | type=product | desc=Google's multimodal foundation model released in 2023.
Candidates:
- id=organization_Gemini | name=Gemini | type=organization | score=0.72 | desc=Cryptocurrency exchange founded in 2014 by the Winklevoss twins.
- id=concept_Gemini | name=Gemini | type=concept | score=0.55 | desc=A zodiac constellation representing twins.
Valid target_existing_id choices: ["organization_Gemini","concept_Gemini"]

EXPECTED OUTPUT (between markers):
===BEGIN===
Gemini||product||ADD||NULL||Gemini||product||Google's multimodal foundation model released in 2023.
===END===

# Case C: Type label differs but SAME real-world entity → UPDATE
[INPUT]
name=OpenAI | type=organization | desc=AI research and deployment company; created GPT models.
Candidates:
- id=org_OpenAI | name=OpenAI | type=org | score=0.91 | desc=An AI company known for foundation models and an API platform.
Valid target_existing_id choices: ["org_OpenAI"]

EXPECTED OUTPUT (between markers):
===BEGIN===
OpenAI||organization||UPDATE||org_OpenAI||OpenAI||organization||An AI research and deployment company known for foundation models and an API platform; creator of GPT models.
===END===

# Case D: Same event TYPE but DIFFERENT subject persons → ADD (SUBJECT GUARD)
[INPUT]
name=check-up with doctor | type=event | desc=Sam visited the doctor for a routine check-up a few days before May 24.
Candidates:
- id=event_checkup_001 | name=check-up with doctor | type=event | score=0.83 | desc=Evan's son had a doctor check-up in early August due to recurring stomach issues.
Valid target_existing_id choices: ["event_checkup_001"]

EXPECTED OUTPUT (between markers):
===BEGIN===
check-up with doctor||event||ADD||NULL||check-up with doctor||event||Sam visited the doctor for a routine check-up a few days before May 24.
===END===
"""
