import joblib
import pandas as pd

MODEL_PATH = "data/models/rendimento_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def predict_rendimento(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return float(prediction)