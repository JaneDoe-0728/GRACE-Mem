"""
Temporal relevance calculation using LiCoMemory-style Weibull decay.
"""
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

from grace_mem.utils.query_time_parser import parse_query_time
from grace_mem.services import Provenance
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retrieval.Temporal", filename="kg_retrieval_temporal.jsonl")


class TemporalRelevanceCalculator:
    """
    Calculate temporal relevance scores using LiCoMemory-style Weibull temporal decay.
    """

    def __init__(self) -> None:
        """Create the temporal relevance helper."""
        pass

    @staticmethod
    def parse_dialogue_datetime(dialogue_datetime: str, request_id: Optional[str] = None) -> Optional[datetime]:
        """
        Parse dialogue_datetime in format: "2023/02/18 (Sat) 08:08"

        Args:
            dialogue_datetime: Datetime string
            request_id: Request ID for logging

        Returns:
            datetime object or None if parsing fails
        """
        if not dialogue_datetime:
            return None

        try:
            dt = parse_query_time(dialogue_datetime)
            if dt is None:
                _jlog("parse_dialogue_datetime_failed", request_id, dialogue_datetime=dialogue_datetime)
            return dt
        except Exception as e:
            _jlog("parse_dialogue_datetime_failed", request_id, error=str(e), dialogue_datetime=dialogue_datetime)
            return None

    @staticmethod
    def get_newest_dialogue_datetime(prov: dict, request_id: Optional[str] = None) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Extract the newest dialogue_datetime from provenance events.

        Args:
            prov: Provenance dictionary
            request_id: Request ID for logging

        Returns:
            (datetime_str, datetime_obj) tuple or (None, None)
        """
        if not prov:
            return None, None

        events = Provenance.prov_to_events(prov)
        if not events:
            return None, None

        # Sort by timestamp descending and get the first one with dialogue_datetime
        sorted_events = sorted(events, key=lambda e: e.get("ts", 0), reverse=True)
        for ev in sorted_events:
            dt_str = ev.get("dialogue_datetime")
            if dt_str:
                dt_obj = TemporalRelevanceCalculator.parse_dialogue_datetime(dt_str, request_id)
                if dt_obj:
                    return dt_str, dt_obj
        return None, None

def date_within_coarse_range(query_date: date, temporal_meta: dict) -> bool:
    """
    Return True when query_date falls within the coarse temporal range of an entity.

    Granularity-aware containment rules:
    - DAY   : exact match (already handled by vector/BM25; kept for symmetry)
    - WEEK  : entity's ISO week contains query_date (±7-day tolerance)
    - MONTH : same year+month
    - SEASON: query_date falls in the season's calendar bounds
    - YEAR  : same year
    - RANGE : query_date is between normalized_start and normalized_end (inclusive)
    """
    if not temporal_meta:
        return False

    granularity = temporal_meta.get("granularity")
    norm_start = temporal_meta.get("normalized_start")
    norm_end = temporal_meta.get("normalized_end")

    if not granularity or not norm_start:
        return False

    try:
        start = date.fromisoformat(norm_start)
    except (ValueError, TypeError):
        return False

    end_str = norm_end or norm_start
    try:
        end = date.fromisoformat(end_str)
    except (ValueError, TypeError):
        end = start

    if granularity == "day":
        return query_date == start

    if granularity == "week":
        # Accept if query_date is within ±7 days of the entity's week span
        return (start - timedelta(days=7)) <= query_date <= (end + timedelta(days=7))

    if granularity == "month":
        return query_date.year == start.year and query_date.month == start.month

    if granularity in ("season",):
        # Use explicit start/end stored by the resolver
        return start <= query_date <= end

    if granularity == "year":
        return query_date.year == start.year

    if granularity == "range":
        return start <= query_date <= end

    return False
