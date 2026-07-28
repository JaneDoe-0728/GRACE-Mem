# Code Review: Retrieval Preamble Generator Implementation

## Overview
Implementation of per-question retrieval preamble generator for auto-attempt-150.

## Files Modified
1. **KG/pipeline/dynamic_planner.py**
   - Added `generate_retrieval_preamble(self, question: str) -> str` method
   - Analyzes question text and generates structured guidance for retrieval agent
   - Uses LLM to produce JSON-formatted preamble with "preamble_text" field

2. **KG/llm/prompts/dynamic_prompting.py**
   - Added `RETRIEVAL_PREAMBLE_SYSTEM` constant
   - Added `RETRIEVAL_PREAMBLE_USER` constant with template variables
   - Templates encode retrieval intent: answer type, entity-relation patterns, constraints

3. **experiment/locomo/stages/qa_eval.py**
   - Modified `rag_answer()` function
   - Calls `generate_retrieval_preamble(query)` before KG context building
   - Injects preamble into system prompt as formatted section

## Validation
- Python syntax: PASSED
- Import check: PASSED
- Branch: `patch/impl-retrieval-preamble-gen-auto-attempt-150`

## Root Cause Addressed
The retrieval agent applies uniform generic strategies to all questions, missing question-specific constraints like:
- Temporal boundaries
- Multi-hop requirements
- Evidence aggregation needs

## Expected Outcome
Question-specific preambles will sharpen retrieval intent, causing the agent to:
- Prioritize evidence matching question-specific patterns
- Respect temporal constraints and time-based filtering
- Follow multi-hop chains when required
- Aggregate evidence for count/listing queries

## Testing
Manual evaluation required to verify:
1. Preamble generation succeeds for various question types
2. Retrieved evidence quality improves for questions with temporal/multi-hop/aggregation constraints
3. Fallback behavior works when preamble generation fails

## Next Steps
- Run auto-attempt-150 evaluation
- Compare retrieval quality metrics vs baseline
- Iterate on prompt templates if needed
