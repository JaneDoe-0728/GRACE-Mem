from string import Formatter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from KG.llm.prompts.keyword.extraction import keyword_extraction_PROMPT


def test_keyword_prompt_placeholders():
    """The prompt takes exactly the two fields Retriever.generate_query_keywords fills.

    ``guidance_section`` was added for the optional retrieval-guidance block; the
    caller renders it as an empty string when there is no guidance. A placeholder
    added here without a matching caller update raises KeyError at query time, so
    the exact set is pinned.
    """
    fields = [
        field_name
        for _, field_name, _, _ in Formatter().parse(keyword_extraction_PROMPT)
        if field_name is not None
    ]

    assert fields == ["query", "guidance_section"]


def test_keyword_prompt_format_preserves_example_json():
    rendered = keyword_extraction_PROMPT.format(
        query="What did Caroline research?", guidance_section=""
    )

    assert 'Query: What did Caroline research?' in rendered
    assert '{"high_level_keywords":' in rendered
    assert '"low_level_keywords":' in rendered


def test_keyword_prompt_renders_the_guidance_block_when_supplied():
    rendered = keyword_extraction_PROMPT.format(
        query="What did Caroline research?",
        guidance_section="\nRetrieval guidance:\nprefer recent sessions\n",
    )

    assert "prefer recent sessions" in rendered
