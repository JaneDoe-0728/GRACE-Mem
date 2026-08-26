"""Keyword extraction prompt for the hybrid retrieval path.

The retriever runs two independent lookups per question and needs different
anchors for each: dense search wants abstract intent ("high_level"), while BM25
and entity/relation matching want literal surface forms ("low_level"). Asking
one LLM call for both keeps the two keyword sets consistent with each other,
which matters because they are later fused by RRF -- if they disagreed about
what the question is asking, fusion would rank the disagreement rather than
the answer.
"""

KEYWORD_EXTRACTION_PROMPT = """
You extract retrieval keywords for memory QA.
Return JSON only with exactly two keys:
- "high_level_keywords": abstract intent or reasoning type
- "low_level_keywords": concrete retrieval anchors from the query

Rules:
- Do not hallucinate missing facts or the final answer.
- low_level_keywords must not be empty.
- low_level_keywords are retrieval clues, not necessarily the answer.
- Keep names, organizations, activities, items, dates, time expressions, and event phrases.
- For first-person questions, ignore pronouns like I/my and extract the actual event, object, or attribute.
- For time/count/comparison questions, include that intent in high_level_keywords.
- Use the retrieval guidance to expand keyword coverage: add entities, attributes, time expressions, and related terms the guidance identifies as retrieval targets, even if they are not explicitly in the query. Reflect the guidance's reasoning type (e.g. temporal comparison, counting, multi-hop, duration) in high_level_keywords.

Example 1:
Query: What did Caroline research?
Retrieval guidance:
- Retrieve all conversation snippets and events where Caroline looked up, studied, or investigated a topic.
- Search raw conversation turns for exact wording of what she researched.
Output: {{"high_level_keywords": ["research topic", "information seeking"], "low_level_keywords": ["Caroline", "research", "looked into", "studied", "investigated"]}}

Example 2:
Query: How many doctor's appointments did I go to in March?
Retrieval guidance:
- Retrieve all events related to doctor visits, medical appointments, and clinic visits in March.
- Deduplicate repeated mentions of the same appointment, then count unique visits.
Output: {{"high_level_keywords": ["count", "deduplication", "medical appointments", "temporal filter"], "low_level_keywords": ["doctor's appointments", "appointments", "clinic", "medical visit", "March"]}}

Example 3:
Query: How long did I wait for the decision on my asylum application?
Retrieval guidance:
- First retrieve the event where the user submitted or filed the asylum application, and confirm its date.
- Then retrieve the event where the decision or result was received, and confirm its date.
- Compute the duration between these two endpoint events by date order.
Output: {{"high_level_keywords": ["duration", "timeline", "multi-hop", "before after"], "low_level_keywords": ["asylum application", "application", "decision", "result", "filed", "submitted", "received"]}}

Example 4:
Query: How many days had passed between the day I bought a gift for my brother's graduation ceremony and the day I bought a birthday gift for my best friend?
Retrieval guidance:
- Retrieve the event where the user bought a gift for the brother's graduation ceremony, and confirm its date.
- Retrieve the event where the user bought a birthday gift for the best friend, and confirm its date.
- Compute the number of days between these two events.
Output: {{"high_level_keywords": ["duration", "day count", "multi-hop", "temporal comparison"], "low_level_keywords": ["gift", "brother", "graduation ceremony", "birthday gift", "best friend", "bought", "purchased"]}}

Query: {query}
{guidance_section}"""
