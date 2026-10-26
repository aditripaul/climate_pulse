"""Configuration precedence: environment variables override the YAML file."""


def test_yaml_values_apply_without_env(fresh_settings, tmp_path):
    conf = tmp_path / "runtime.yaml"
    conf.write_text("model:\n  max_epochs: 7\nnwm:\n  conditioning: forecast\n")
    s = fresh_settings(RUNTIME_CONF=conf)
    assert s.model.max_epochs == 7
    assert s.nwm.conditioning == "forecast"


def test_env_overrides_yaml(fresh_settings, tmp_path):
    conf = tmp_path / "runtime.yaml"
    conf.write_text("model:\n  max_epochs: 7\n  seed: 1\nnwm:\n  area_weighted_loss: true\n")
    s = fresh_settings(
        RUNTIME_CONF=conf, MODEL_MAX_EPOCHS=3, MODEL_SEED=9, NWM_AREA_WEIGHTED_LOSS="false"
    )
    assert s.model.max_epochs == 3
    assert s.model.seed == 9
    assert s.nwm.area_weighted_loss is False
