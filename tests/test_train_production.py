import pandas as pd

from ml.train_production import (
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    build_pipeline,
    build_metrics,
    load_real_data,
)


def test_production_paths_and_columns_are_defined():
    assert DATA_PATH.as_posix() == "data/processed/pnad_real_processed.csv"
    assert MODEL_PATH.as_posix() == "data/models/rendimento_model_production.pkl"
    assert METRICS_PATH.as_posix() == "data/models/metrics_production.json"
    assert FEATURES == ["idade", "sexo", "cor_raca", "anos_estudo", "setor", "regiao"]
    assert TARGET == "rendimento_hora"


def test_load_real_data_reads_csv_from_custom_path(tmp_path):
    csv_path = tmp_path / "pnad_real_processed.csv"
    expected_df = pd.DataFrame(
        {
            "idade": [35],
            "sexo": ["M"],
            "cor_raca": ["Parda"],
            "anos_estudo": [12],
            "setor": ["Comercio"],
            "regiao": ["Sudeste"],
            "rendimento_hora": [20.0],
        }
    )
    expected_df.to_csv(csv_path, index=False)

    loaded_df = load_real_data(csv_path)

    pd.testing.assert_frame_equal(loaded_df, expected_df)


def test_build_pipeline_uses_expected_steps_and_xgboost_params():
    pipeline = build_pipeline()
    model = pipeline.named_steps["model"]

    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert model.n_estimators == 200
    assert model.max_depth == 5
    assert model.learning_rate == 0.08
    assert model.objective == "reg:squarederror"
    assert model.random_state == 42


def test_build_metrics_includes_prediction_interval_residuals():
    metrics = build_metrics(
        rmse=5.86,
        mae=4.34,
        r2=0.333,
        n_rows=175132,
        lower_residual_p05=-8.25,
        upper_residual_p95=9.75,
    )

    assert metrics["prediction_interval_method"] == "residual_percentile_5_95"
    assert metrics["prediction_interval_residuals"] == {
        "lower_residual_p05": -8.25,
        "upper_residual_p95": 9.75,
    }
