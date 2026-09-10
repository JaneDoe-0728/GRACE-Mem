"""Compressor: llmlingua-based turn summarization."""
import time
from typing import Any

from grace_mem.temporal import TimeContext, rewrite_temporal_text
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")


class Compressor:
    """Wraps llmlingua PromptCompressor and writes summaries to the VDB."""

    def __init__(self, *, summaries_vdb: Any) -> None:
        """Store the summaries vector store and lazily loaded compressor."""
        self._summaries_vdb = summaries_vdb
        self._prompt_compressor = None

    def _get_compressor(self) -> Any:
        """Create the llmlingua compressor on first use and then reuse it."""
        if self._prompt_compressor is None:
            from llmlingua import PromptCompressor as _PromptCompressor
            self._prompt_compressor = _PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
            )
        return self._prompt_compressor

    def summarize_turn(
        self,
        session_id: int | str,
        message_id: int,
        user_text: str,
        assistant_text: str,
        request_id: str,
        dialogue_datetime: str | None = None,
        temporal_hints: list | None = None,
        tctx: TimeContext | None = None,
    ) -> tuple[str, str]:
        """Compress current turn, write to summaries VDB; returns (summary_id, summary_text)."""
        if not assistant_text or not assistant_text.strip():
            curr_text = user_text.strip()
        else:
            curr_text = f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"

        start_time = time.time()
        try:
            results = self._get_compressor().compress_prompt_llmlingua2(
                curr_text,
                rate=0.6,
                force_tokens=['\n', '.', '!', '?', ','],
                chunk_end_tokens=['.', '\n'],
                return_word_label=False,
                drop_consecutive=True,
            )
            summary = results.get("compressed_prompt", "").strip() or curr_text
            compression_rate = results.get("rate", None)
            elapsed = time.time() - start_time
            _jlog(
                "prompt_compressed",
                request_id,
                compression_rate=compression_rate,
                compression_time=f"{elapsed:.2f}s",
                original_length=len(curr_text),
                compressed_length=len(summary),
            )
        except Exception as e:
            summary = curr_text
            _jlog("prompt_compress_failed", request_id, error=str(e), fallback_length=len(summary))

        # Rewrite relative temporal phrases in the stored summary to resolved values.
        if temporal_hints and tctx is not None:
            rewritten_summary, _ = rewrite_temporal_text(summary, tctx)
            summary = rewritten_summary

        summary_id = self._summaries_vdb.add_summary(
            session_id, message_id, summary, dialogue_datetime=dialogue_datetime,
            raw_text=curr_text,
        )
        _jlog("summary_added_to_vdb", request_id, summary_id=summary_id, summary_length=len(summary))
        return summary_id, summary
