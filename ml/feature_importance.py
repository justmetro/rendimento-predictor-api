import joblib


PRODUCTION_MODEL_PATH = "data/models/rendimento_model.pkl"
REAL_MODEL_PATH = "data/models/rendimento_model_real.pkl"


def clean_feature_name(feature_name: str) -> str:
    return (
        feature_name
        .replace("cat__", "")
        .replace("num__", "")
    )


def get_feature_importance(model_path: str, top_n: int = 15) -> list[dict]:
    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    feature_importance = [
        {
            "feature": clean_feature_name(feature_name),
            "importance": round(float(importance), 6),
        }
        for feature_name, importance in zip(feature_names, importances)
    ]

    feature_importance = sorted(
        feature_importance,
        key=lambda item: item["importance"],
        reverse=True,
    )

    return feature_importance[:top_n]


def get_production_feature_importance() -> dict:
    return {
        "model_name": "random_forest_sintetico_v1",
        "model_type": "production",
        "feature_importance": get_feature_importance(PRODUCTION_MODEL_PATH),
    }


def get_real_feature_importance() -> dict:
    return {
        "model_name": "random_forest_real_v1",
        "model_type": "candidate",
        "feature_importance": get_feature_importance(REAL_MODEL_PATH),
    }