# System Design: Retrieval Preamble Generator

## Overview
Per-question retrieval preamble generation system that analyzes question text and injects targeted guidance into the agent system prompt before knowledge graph retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Query Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Question → Question Analyzer → Preamble Generator          │
│                        ↓                                          │
│              Structured Preamble (JSON)                           │
│                        ↓                                          │
│         Injected into Agent System Prompt                         │
│                        ↓                                          │
│           KG Retrieval (with targeted guidance)                   │
│                        ↓                                          │
│              Answer Generation                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. DynamicPlanner.generate_retrieval_preamble()
**Location**: `KG/pipeline/dynamic_planner.py:359-416`

**Input**: Question string

**Output**: Natural-language preamble text or empty string on failure

**Internal Steps**:
1. Classify question type (single_hop, multi_hop, temporal, aggregation)
2. Analyze question structure (entity-relation patterns, constraints)
3. Build structured data object with analysis results
4. Call LLM with RETRIEVAL_PREAMBLE_SYSTEM + RETRIEVAL_PREAMBLE_USER prompts
5. Parse JSON response and extract "preamble_text" field
6. Return formatted guidance string

### 2. Prompt Templates

**RETRIEVAL_PREAMBLE_SYSTEM** (`KG/llm/prompts/dynamic_prompting.py:178-207`):
- Defines agent role as retrieval guidance generator
- Specifies analysis focus areas
- Enumerates question types
- Enforces JSON output format

**RETRIEVAL_PREAMBLE_USER** (`KG/llm/prompts/dynamic_prompting.py:209-219`):
- Receives structured analysis from DynamicPlanner
- Template variables: question, question_type, retrieval_focus, entity_relation_pattern, key_entities, relationship_types, retrieval_constraints, is_multi_hop, is_count_query, is_comparison_query
- Requests guidance that elevates evidence matching question-specific patterns

### 3. qa_eval Integration
**Location**: `experiment/locomo/stages/qa_eval.py:155-166`

**Flow**:
1. Calls `generate_retrieval_preamble(query)` if dynamic prompting enabled
2. Wraps preamble in formatted section marker
3. Injects into agent system prompt before KG context
4. Fallback to empty string on generation failure

## Question Type Classification

| Type | Description | Example |
|------|-------------|---------|
| single_hop | Direct entity-attribute lookup | "What is Emily's birth date?" |
| multi_hop | Requires traversing entity relationships | "Who are Emily's friends who live in Seattle?" |
| temporal | Requires time-based filtering | "What did Emily do last week?" |
| aggregation | Requires counting, summing, or listing | "How many meals did Emily eat this month?" |

## Evidence Prioritization Patterns

The preamble guides retrieval to prioritize evidence containing:
1. **Entity-relation matches**: Gold summaries likely contain specific entity-relation pairs
2. **Temporal constraints**: Time-based filtering for date/timespan qualifiers
3. **Multi-hop chains**: Relationship traversal paths for connected entity queries
4. **Aggregation indicators**: Count/listing requirements for summary generation

## Fallback Strategy

If preamble generation fails:
- Return empty string
- Continue with generic retrieval guidance (existing behavior)
- Log error to console
- No impact on answer generation flow

## Validation Checklist

- [x] Python syntax validation passed
- [x] Import check passed
- [x] Branch created: `patch/impl-retrieval-preamble-gen-auto-attempt-150`
- [x] Manifest JSON written
- [x] Research log entry appended
- [ ] Auto-attempt-150 evaluation run
- [ ] Retrieval quality metrics compared vs baseline

## Future Enhancements

1. **Preamble caching**: Cache results for identical questions
2. **Prompt iteration**: A/B test different template formulations
3. **Classification refinement**: Improve question type accuracy
4. **Feedback loop**: Use answer quality to retrain preamble generation
5. **Multi-modal support**: Extend to images, documents, other KG types
