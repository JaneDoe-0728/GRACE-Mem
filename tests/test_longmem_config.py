from dataclasses import FrozenInstanceError

import pytest

from experiment.experiment_config import INGEST_PARAMS, RETRIEVAL_PARAMS
from experiment.longmem.models import DatasetConfig


def test_dataset_config_accepts_shared_params_without_locomo_only_keys():
    config = DatasetConfig.from_params(
        name="dataset-1",
        csv_path="dataset-1.csv",
        ingest_params=INGEST_PARAMS,
        retrieval_params=RETRIEVAL_PARAMS,
    )

    assert config.use_split_summary is INGEST_PARAMS["use_split_summary"]
    assert config.ent_topk == RETRIEVAL_PARAMS["ent_topk"]
    assert "chunk_turns" not in config.__dataclass_fields__
    assert config.retrieval_kwargs()["summary_topk_per_item"] == 16


def test_dataset_config_is_immutable():
    config = DatasetConfig.from_params(
        name="dataset-1",
        csv_path="dataset-1.csv",
        ingest_params=INGEST_PARAMS,
        retrieval_params=RETRIEVAL_PARAMS,
    )

    with pytest.raises(FrozenInstanceError):
        config.ent_topk = 99


def test_dataset_config_rejects_invalid_threshold():
    retrieval_params = {**RETRIEVAL_PARAMS, "ent_threshold": 1.5}

    with pytest.raises(ValueError, match="ent_threshold must be between 0 and 1"):
        DatasetConfig.from_params(
            name="dataset-1",
            csv_path="dataset-1.csv",
            ingest_params=INGEST_PARAMS,
            retrieval_params=retrieval_params,
        )
