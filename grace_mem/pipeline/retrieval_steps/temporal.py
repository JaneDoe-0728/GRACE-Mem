"""
Temporal relevance calculation using LiCoMemory-style Weibull decay.
"""
import math
import statistics
from datetime import date, datetime, timedelta
from typing import Optional, Set, Dict, Tuple

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

    def collect_time_deltas(
        self,
        query_time_dt: datetime,
        entity_ids: Set[str],
        relationship_ids: Set[str],
        entity_id2meta: Dict[str, dict],
        relationship_id2meta: Dict[str, dict],
        request_id: Optional[str] = None,
    ) -> list[float]:
        """
        Collect time deltas (query_time - dialogue_datetime) in days for all items.

        Args:
            query_time_dt: Query datetime
            entity_ids: Set of entity IDs
            relationship_ids: Set of relationship IDs
            entity_id2meta: Entity metadata map
            relationship_id2meta: Relationship metadata map
            request_id: Request ID for logging

        Returns:
            List of non-negative Δτ values in days
        """
        deltas = []

        # From entities
        for entity_id in entity_ids:
            prov = entity_id2meta.get(entity_id, {}).get("prov", {})
            _, dt_obj = self.get_newest_dialogue_datetime(prov, request_id)
            if dt_obj:
                delta_days = (query_time_dt - dt_obj).total_seconds() / 86400
                if delta_days >= 0:  # Ignore future timestamps
                    deltas.append(delta_days)

        # From relationships
        for relationship_id in relationship_ids:
            prov = relationship_id2meta.get(relationship_id, {}).get("prov", {})
            _, dt_obj = self.get_newest_dialogue_datetime(prov, request_id)
            if dt_obj:
                delta_days = (query_time_dt - dt_obj).total_seconds() / 86400
                if delta_days >= 0:  # Ignore future timestamps
                    deltas.append(delta_days)

        return deltas

    def calculate_adaptive_scale(
        self,
        time_deltas: list[float],
        min_tau: float = 1.0,
        max_tau: float = 90.0,
        request_id: Optional[str] = None,
    ) -> Optional[float]:
        """
        Calculate median-based adaptive time scale (tau_hat) from time deltas.

        Args:
            time_deltas: List of time deltas in days
            min_tau: Minimum tau value
            max_tau: Maximum tau value
            request_id: Request ID for logging

        Returns:
            tau_hat value or None if no valid deltas
        """
        if not time_deltas:
            _jlog("temporal_weighting_disabled", request_id, reason="no_valid_deltas")
            return None

        # LiCoMemory: tau_hat = median of all Δτ values
        tau_hat = statistics.median(time_deltas)

        # Clamp tau_hat to reasonable range
        tau_hat = max(min_tau, min(max_tau, tau_hat))

        _jlog(
            "temporal_adaptive_scale",
            request_id,
            delta_count=len(time_deltas),
            tau_hat=tau_hat,
            min_delta=min(time_deltas),
            max_delta=max(time_deltas),
            median_delta=statistics.median(time_deltas),
        )

        return tau_hat

    @staticmethod
    def compute_temporal_weight(
        query_time_dt: Optional[datetime],
        item_datetime: Optional[datetime],
        tau_hat: Optional[float],
        k: float = 0.5,
        request_id: Optional[str] = None,
    ) -> float:
        """
        LiCoMemory-style Weibull temporal decay with long-tail behavior.

        w(Δτ) = exp(- (Δτ / tau_hat) ** k), where k < 1

        Args:
            query_time_dt: Query datetime
            item_datetime: Item datetime
            tau_hat: Adaptive time scale
            k: Weibull shape parameter (default 0.5 for long tail)
            request_id: Request ID for logging

        Returns:
            Temporal weight in (0, 1], or 1.0 if temporal weighting disabled
        """
        if not query_time_dt or not tau_hat or not item_datetime:
            return 1.0  # No temporal modulation

        try:
            # Δτ = query_time - dialogue_datetime (in days)
            delta_tau = (query_time_dt - item_datetime).total_seconds() / 86400

            if delta_tau < 0:
                # Future timestamp: return neutral weight
                return 1.0

            # Weibull decay: w(Δτ) = exp(- (Δτ / tau_hat) ** k)
            weight = math.exp(-((delta_tau / tau_hat) ** k))

            # Clamp to (0, 1]
            weight = max(1e-6, min(1.0, weight))

            return weight

        except Exception as e:
            _jlog("temporal_weight_calculation_failed", request_id, error=str(e))
            return 1.0


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
