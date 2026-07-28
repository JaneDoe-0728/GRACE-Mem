# llm/prompts/keyword/extraction.py
# """
# Keyword extraction prompt for hybrid retrieval.
# Extracts high-level (concepts) and low-level (specific entities) keywords.
# """

# keyword_extraction_PROMPT = (
#     "---Role---\n\n"
#     "You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.\n\n"
#     "---Goal---\n\n"
#     "Given the query and conversation history, list both high-level and low-level keywords. "
#     "High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.\n\n"
#     "---Instructions---\n\n"
#     "- Consider both the current query and relevant conversation history when extracting keywords\n"
#     "- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output\n"
#     "- The JSON should have two keys:\n"
#     "  - \"high_level_keywords\" for overarching concepts or themes\n"
#     "  - \"low_level_keywords\" for specific entities or details\n\n"
#     "######################\n---Examples---\n######################\n"
#     "Example 1:\n\n"
#     "Query: \"How does international trade influence global economic stability?\"\n"
#     "################\n"
#     "Output:\n"
#     "{{\n"
#     "  \"high_level_keywords\": [\"International trade\", \"Global economic stability\", \"Economic impact\"],\n"
#     "  \"low_level_keywords\": [\"Trade agreements\", \"Tariffs\", \"Currency exchange\", \"Imports\", \"Exports\"]\n"
#     "}}\n"
#     "#############################\n"
#     "Example 2:\n\n"
#     "Query: \"What are the environmental consequences of deforestation on biodiversity?\"\n"
#     "################\n"
#     "Output:\n"
#     "{{\n"
#     "  \"high_level_keywords\": [\"Environmental consequences\", \"Deforestation\", \"Biodiversity loss\"],\n"
#     "  \"low_level_keywords\": [\"Species extinction\", \"Habitat destruction\", \"Carbon emissions\", \"Rainforest\", \"Ecosystem\"]\n"
#     "}}\n"
#     "#############################\n"
#     "Example 3:\n\n"
#     "Query: \"What is the role of education in reducing poverty?\"\n"
#     "################\n"
#     "Output:\n"
#     "{{\n"
#     "  \"high_level_keywords\": [\"Education\", \"Poverty reduction\", \"Socioeconomic development\"],\n"
#     "  \"low_level_keywords\": [\"School access\", \"Literacy rates\", \"Job training\", \"Income inequality\"]\n"
#     "}}\n"
#     "#############################\n\n"
#     "---Real Data---\n"
#     "######################\n"
#     "Query: {query}\n"
#     "######################\n"
#     "Output:"
# )

"""
Keyword extraction prompt for hybrid retrieval.
Extracts high-level (concepts) and low-level (specific entities) keywords.
"""

# 
keyword_extraction_PROMPT = """
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
