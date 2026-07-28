"""
Test script to evaluate reranker performance on entities and relationships.

This script allows you to test how the reranker scores different entities and relationships
against a given question, helping you tune threshold and topk parameters.

FEATURES:
1. Test with hardcoded sample entities/relationships
2. Test with specific IDs from your artifacts directory
3. Auto-loads metadata from artifacts_00ca467f
4. Shows scores and threshold analysis

USAGE:

1. Basic usage (test with samples):
    python test_reranker.py

2. Test your own entities/relationships:
    Edit main() function and set:
    - TEST_QUESTION: Your question
    - TEST_ENTITY_IDS: List of entity IDs to test
    - TEST_REL_IDS: List of relationship IDs to test
    - ARTIFACTS_DIR: Path to your artifacts directory (default: artifacts_00ca467f)

3. Example configuration in main():
    TEST_QUESTION = "How many doctor's appointments did I go to in March?"
    TEST_ENTITY_IDS = [
        "person_dr._thompson",
        "event_orthopedic_surgeon_appointment",
        "time_march",
    ]
    TEST_REL_IDS = [
        "person_user_event_wudhu",
        "event_wudhu_concept_qiblah",
    ]

4. Choose reranker mode:
    RERANKER_MODE = "v1"  # CrossEncoder (fast, uses similarity scores)
    RERANKER_MODE = "v2"  # LLM pointwise (slower, uses Yes/No logit difference)
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from KG.utils.reranker import get_reranker
from AI.KG.utils.reranker import get_reranker_v2


# Configuration: Set your artifacts directory and test data here
# ARTIFACTS_DIR = "experiment/multi_dataset_output/artifacts_00ca467f"
ARTIFACTS_DIR = "experiment/multi_dataset_output/artifacts_10d9b85a"

# Reranker mode: "v1" (CrossEncoder) or "v2" (LLM pointwise Yes/No logits)
RERANKER_MODE = "v1"  # Change to "v1" to use CrossEncoder mode


def get_reranker_by_mode(mode: str = RERANKER_MODE):
    """Get reranker instance based on mode."""
    if mode == "v2":
        print(f"[Using Reranker V2] LLM pointwise reranking (Yes/No logits)")
        return get_reranker_v2()
    else:
        print(f"[Using Reranker V1] CrossEncoder reranking")
        return get_reranker()


def load_entity_metas_from_artifacts(entity_ids: List[str], artifacts_dir: str = ARTIFACTS_DIR) -> List[Dict[str, Any]]:
    """
    Load entity metadata from artifacts directory by entity IDs.

    Args:
        entity_ids: List of entity IDs to fetch
        artifacts_dir: Path to artifacts directory

    Returns:
        List of entity metadata dictionaries
    """
    meta_path = Path(artifacts_dir) / "entities_meta.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Entity metadata not found: {meta_path}")

    # Load all metas and filter by ID
    entity_metas = []
    entity_ids_set = set(entity_ids)

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta = json.loads(line)
                if meta.get("id") in entity_ids_set:
                    entity_metas.append(meta)

    # Sort by the order in entity_ids
    id_to_meta = {m["id"]: m for m in entity_metas}
    sorted_metas = [id_to_meta[eid] for eid in entity_ids if eid in id_to_meta]

    print(f"✓ Loaded {len(sorted_metas)}/{len(entity_ids)} entities from {meta_path}")

    missing = [eid for eid in entity_ids if eid not in id_to_meta]
    if missing:
        print(f"⚠ Missing entity IDs: {missing}")

    return sorted_metas


def load_relationship_metas_from_artifacts(rel_ids: List[str], artifacts_dir: str = ARTIFACTS_DIR) -> List[Dict[str, Any]]:
    """
    Load relationship metadata from artifacts directory by relationship IDs.

    Args:
        rel_ids: List of relationship IDs to fetch
        artifacts_dir: Path to artifacts directory

    Returns:
        List of relationship metadata dictionaries
    """
    meta_path = Path(artifacts_dir) / "relationships_meta.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Relationship metadata not found: {meta_path}")

    # Load all metas and filter by ID
    relationship_metas = []
    rel_ids_set = set(rel_ids)

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta = json.loads(line)
                if meta.get("id") in rel_ids_set:
                    relationship_metas.append(meta)

    # Sort by the order in rel_ids
    id_to_meta = {m["id"]: m for m in relationship_metas}
    sorted_metas = [id_to_meta[rid] for rid in rel_ids if rid in id_to_meta]

    print(f"✓ Loaded {len(sorted_metas)}/{len(rel_ids)} relationships from {meta_path}")

    missing = [rid for rid in rel_ids if rid not in id_to_meta]
    if missing:
        print(f"⚠ Missing relationship IDs: {missing}")

    return sorted_metas


def test_entities_from_artifacts(
    question: str,
    entity_ids: List[str],
    artifacts_dir: str = ARTIFACTS_DIR
):
    """
    Test reranking on entities loaded from artifacts directory.

    Args:
        question: The question to test against
        entity_ids: List of entity IDs to test
        artifacts_dir: Path to artifacts directory
    """
    print("\n\n" + "=" * 80)
    print("TESTING ENTITIES FROM ARTIFACTS")
    print("=" * 80)
    print(f"Artifacts: {artifacts_dir}")
    print(f"Question: {question}\n")

    # Load entity metas from artifacts
    entity_metas = load_entity_metas_from_artifacts(entity_ids, artifacts_dir)

    if not entity_metas:
        print("⚠ No entities found. Check your entity IDs and artifacts path.")
        return

    # Build entity texts (same format as EntityManager)
    entity_texts = []
    for meta in entity_metas:
        name = meta.get("name", "").strip()
        type_val = meta.get("type", "").strip()
        description = meta.get("description", "").strip()

        if name:
            text = f"{name} [type={type_val}] {description}".strip()
        else:
            text = f"{type_val} {description}".strip()
        entity_texts.append(text)

    # Initialize reranker
    print("\nLoading reranker model...")
    reranker = get_reranker_by_mode(RERANKER_MODE)
    print("✓ Reranker loaded\n")

    # Rank entities
    print("-" * 80)
    print("RERANKING RESULTS:")
    print("-" * 80)

    results = reranker.rank_pairs(
        query=question,
        texts=entity_texts,
        threshold=None  # Show all scores
    )

    # Display results
    for rank, (idx, score) in enumerate(results, 1):
        meta = entity_metas[idx]
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   ID: {meta['id']}")
        print(f"   Name: {meta.get('name', 'N/A')}")
        print(f"   Type: {meta.get('type', 'N/A')}")
        print(f"   Description: {meta.get('description', 'N/A')[:100]}")

        # Show if it would be recovered at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        recovered = [f"{t:.1f}✓" if score >= t else f"{t:.1f}✗" for t in thresholds]
        print(f"   Recovery @ thresholds: {' | '.join(recovered)}")

    # Threshold analysis
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        filtered_results = [(idx, score) for idx, score in results if score >= threshold]
        print(f"\nThreshold {threshold:.1f}: {len(filtered_results)} entities would be recovered")
        for idx, score in filtered_results[:3]:
            print(f"  - {entity_metas[idx].get('name', entity_metas[idx]['id'])} ({score:.4f})")


def test_relationships_from_artifacts(
    question: str,
    rel_ids: List[str],
    artifacts_dir: str = ARTIFACTS_DIR
):
    """
    Test reranking on relationships loaded from artifacts directory.

    Args:
        question: The question to test against
        rel_ids: List of relationship IDs to test
        artifacts_dir: Path to artifacts directory
    """
    print("\n\n" + "=" * 80)
    print("TESTING RELATIONSHIPS FROM ARTIFACTS")
    print("=" * 80)
    print(f"Artifacts: {artifacts_dir}")
    print(f"Question: {question}\n")

    # Load relationship metas from artifacts
    relationship_metas = load_relationship_metas_from_artifacts(rel_ids, artifacts_dir)

    if not relationship_metas:
        print("⚠ No relationships found. Check your relationship IDs and artifacts path.")
        return

    # Build relationship texts (same format as RelationshipManager)
    relationship_texts = []
    for meta in relationship_metas:
        source_entity = meta.get("source_entity", "")
        target_entity = meta.get("target_entity", "")
        description = meta.get("description", "")
        keywords = meta.get("keywords", "")

        text = f"{source_entity} -> {target_entity} | {description} (keywords: {keywords})"
        relationship_texts.append(text)

    # Initialize reranker
    print("\nLoading reranker model...")
    reranker = get_reranker_by_mode(RERANKER_MODE)
    print("✓ Reranker loaded\n")

    # Rank relationships
    print("-" * 80)
    print("RERANKING RESULTS:")
    print("-" * 80)

    results = reranker.rank_pairs(
        query=question,
        texts=relationship_texts,
        threshold=None  # Show all scores
    )

    # Display results
    results = results
    for rank, (idx, score) in enumerate(results, 1):
        meta = relationship_metas[idx]
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   ID: {meta['id']}")
        print(f"   Relationship: {meta.get('source_entity', '?')} -> {meta.get('target_entity', '?')}")
        print(f"   Description: {meta.get('description', 'N/A')[:100]}")
        print(f"   Keywords: {meta.get('keywords', 'N/A')}")

    #     # Show if it would be recovered at different thresholds
    #     thresholds = [0.6, 0.8]
    #     recovered = [f"{t:.1f}✓" if score >= t else f"{t:.1f}✗" for t in thresholds]
    #     print(f"   Recovery @ thresholds: {' | '.join(recovered)}")

    # # Threshold analysis
    # print("\n" + "=" * 80)
    # print("THRESHOLD ANALYSIS")
    # print("=" * 80)

    # for threshold in [0.6, 0.8]:
    #     filtered_results = [(idx, score) for idx, score in results if score >= threshold]
    #     print(f"\nThreshold {threshold:.1f}: {len(filtered_results)} relationships would be recovered")
    #     for idx, score in filtered_results[:8]:
    #         meta = relationship_metas[idx]
    #         print(f"  - {meta.get('source_entity', '?')} -> {meta.get('target_entity', '?')} ({score:.4f})")


def main():
    """Run all tests"""
    try:
        # ========================================================================
        # CONFIGURE YOUR TEST HERE
        # ========================================================================

        RUN_ARTIFACTS_TESTS = True

        # Configure your test data here:
        # TEST_QUESTION = "How many doctor's appointments did I go to in March?"
        TEST_QUESTION = "How many days did I spend attending workshops, lectures, and conferences in April?"

        # TEST_ENTITY_IDS = [
        #     "person_dr._thompson",
        #     "event_orthopedic_surgeon_appointment",
        #     "person_user",
        #     "time_march",
        #     # Add more entity IDs here...
        # ]
        # TEST_REL_IDS = [
        #     "event_madrid_train_bombings_location_spain",
        #     "activity_in_depth_follow_up_date_2023-03-27",
        #     "concept_engagement_rate_concept_total_reach",
        #     "event_gender_pay_gap_statistics_provision_date_2023-03-27",
        #     "person_user_activity_post_major_update_tracking",
        #     "event_madrid_train_bombings_person_osama_bin_laden",
        #     "event_clean_bowls_event_timespan_2023-03-20_to_2023-03-26",
        #     "event_numbness_discussion_person_dr._smith",
        #     "activity_send_brief_personalized_email_date_2023-03-27",
        #     "event_speed_networking_session_date_2023-03-27",
        #     "person_user_product_google_analytics",
        #     "product_fitbit_scale_topic_fitness_tracking",
        #     "person_dr._smith_service_ct_scan",
        #     "event_mom's_50th_birthday_party_date_2023-02",
        #     "event_numbness_discussion_person_dr._johnson",
        #     "person_user_topic_fitness_tracking",
        #     "event_madrid_train_bombings_organization_al-qaeda",
        #     "topic_decline_print_readership_concept_print_media_industries",
        #     "event_appointment_with_dr._patel_person_dr._patel",
        #     "event_orthopedic_surgeon_appointment_person_dr._thompson",
        #     "event_madrid_train_bombings_date_2004-03-11",
        #     "product_samsung_smartwatch_topic_fitness_tracking",
        #     "event_baseball_game_(february_2023)_date_2023-02",
        # ]

        TEST_REL_IDS = [
            "event_attended_lecture_location_public_library",
            "person_user_topic_time_management_tools",
            "event_customized_meal_plan_event_topic_busy_day_strategies",
            "event_customized_meal_plan_event_topic_special_occasion_strategies",
            "person_user_product_headspace",
            "person_user_person_volunteer_coordinator",
            "person_user_event_workshop",
            "topic_depression_person_user",
            "event_birthday_last_month_timespan_2023-04",
            "event_workshop_concept_feature_engineering",
            "person_user_event_attended_lecture",
            "person_user_activity_class_or_workshop",
            "person_user_product_todoist",
            "product_broken_horses_date_april_2021",
            "person_user_activity_morning_walks",
            "person_user_event_workshops_classes_with_teacher",
            "person_user_product_rescuetime",
            "person_user_product_bright_hour_memoir_of_living_and_dying",
            "person_user_event_morning_walk",
            "person_user_activity_game_nights",
            "person_user_topic_music_production_podcast",
            "person_user_service_meetup.com",
            "topic_anxiety_person_user",
            "person_user_concept_hobby_based_meetups",
            "concept_current_daily_step_count_timespan_2023-04-27_to_2023-04-30",
            "person_user_event_conversation_at_events_meetups",
            "person_user_activity_meetups",
            "event_hrv_increase_timespan_2023-04-03_to_2023-04-30",
            "event_attended_lecture_date_2023-04-10",
            "person_user_event_studio_tour",
            "event_average_heart_rate_decrease_timespan_2023-04-03_to_2023-04-30",
            "event_order_travel_adapter_date_2023-04-17",
            "product_crying_in_h_mart_date_april_2021",
            "event_ml_workshop_2day_april_date_2023-04-18",
            "event_spring_sales_timespan_2023-03_to_2023-04",
            "event_heart_rate_recovery_improvement_timespan_2023-04-03_to_2023-04-30",
            "person_user_event_search_award_seats",
            "person_user_event_meaningful_in-person_meetup",
            "event_winter_farmers_market_timespan_saturdays,_december_to_april",
            "event_watch_episode_date_2023-04-30",
            "event_prepare_breakfast_sunday_evening_date_2023-04-30"
        ]

        # ========================================================================
        # RUN TESTS
        # ========================================================================

        if RUN_ARTIFACTS_TESTS:
            # Test 3: Entities from artifacts
            # if TEST_ENTITY_IDS:
            #     test_entities_from_artifacts(
            #         question=TEST_QUESTION,
            #         entity_ids=TEST_ENTITY_IDS,
            #         artifacts_dir=ARTIFACTS_DIR
            #     )

            # Test 4: Relationships from artifacts
            if TEST_REL_IDS:
                test_relationships_from_artifacts(
                    question=TEST_QUESTION,
                    rel_ids=TEST_REL_IDS,
                    artifacts_dir=ARTIFACTS_DIR
                )

        print("\n" + "=" * 80)
        print("✓ All tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
