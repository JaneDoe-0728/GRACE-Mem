from string import Formatter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from KG.llm.prompts.keyword.extraction import keyword_extraction_PROMPT


def test_keyword_prompt_only_uses_query_placeholder():
    fields = [
        field_name
        for _, field_name, _, _ in Formatter().parse(keyword_extraction_PROMPT)
        if field_name is not None
    ]

    assert fields == ["query"]


def test_keyword_prompt_format_preserves_example_json():
    rendered = keyword_extraction_PROMPT.format(query="What did Caroline research?")

    assert 'Query: What did Caroline research?' in rendered
    assert '{"high_level_keywords":' in rendered
    assert '"low_level_keywords":' in rendered
