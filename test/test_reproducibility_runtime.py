from KG.runtime.reproducibility import resolve_reproducibility_config
from experiment.experiment_config import REPRODUCIBILITY_PARAMS
from experiment.common.reproducibility import (
    ReproducibilityConfig,
    resolve_reproducibility_config as resolve_experiment_config,
)


def test_core_runtime_resolves_explicit_environment_and_outer_defaults(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_SEED", raising=False)
    monkeypatch.delenv("EXPERIMENT_DETERMINISTIC", raising=False)

    defaults = resolve_reproducibility_config(
        defaults={"seed": 7, "deterministic": False}
    )
    assert defaults == ReproducibilityConfig(seed=7, deterministic=False)

    monkeypatch.setenv("EXPERIMENT_SEED", "11")
    monkeypatch.setenv("EXPERIMENT_DETERMINISTIC", "true")
    inherited = resolve_reproducibility_config(
        defaults={"seed": 7, "deterministic": False}
    )
    assert inherited == ReproducibilityConfig(seed=11, deterministic=True)


def test_experiment_adapter_uses_experiment_config(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_SEED", raising=False)
    monkeypatch.delenv("EXPERIMENT_DETERMINISTIC", raising=False)

    config = resolve_experiment_config()

    assert config.seed == REPRODUCIBILITY_PARAMS["seed"]
    assert config.deterministic == REPRODUCIBILITY_PARAMS["deterministic"]
