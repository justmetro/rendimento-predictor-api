def get_faixa_etaria(idade: int) -> str:
    if idade < 18:
        return "14-17"
    if idade < 25:
        return "18-24"
    if idade < 35:
        return "25-34"
    if idade < 45:
        return "35-44"
    if idade < 55:
        return "45-54"
    if idade < 65:
        return "55-64"
    return "65+"


def get_escolaridade(anos_estudo: int) -> str:
    if anos_estudo == 0:
        return "Sem instrucao"
    if anos_estudo <= 8:
        return "Fundamental incompleto"
    if anos_estudo <= 11:
        return "Fundamental completo ou medio incompleto"
    if anos_estudo == 12:
        return "Medio completo"
    if anos_estudo <= 15:
        return "Superior incompleto"
    return "Superior completo ou mais"


def anonymize_history_response(response: dict) -> dict:
    anonymized_predictions = []

    for prediction in response.get("predictions", []):
        anonymized_prediction = dict(prediction)
        idade = anonymized_prediction.pop("idade")
        anonymized_prediction.pop("sexo", None)
        anonymized_prediction.pop("cor_raca", None)

        anonymized_prediction["faixa_etaria"] = get_faixa_etaria(idade)
        anonymized_prediction["escolaridade"] = get_escolaridade(
            anonymized_prediction["anos_estudo"]
        )
        anonymized_predictions.append(anonymized_prediction)

    return {
        **response,
        "predictions": anonymized_predictions,
    }
