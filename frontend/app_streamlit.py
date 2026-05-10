import os

import requests
import streamlit as st


PRODUCTION_API_URL = "https://rendimento-predictor-api.onrender.com"
LOCAL_API_URL = "http://localhost:8000"
FALLBACK_METADATA = {
    "numeric_constraints": {
        "idade": {
            "min": 14,
            "max": 100,
        },
        "anos_estudo": {
            "min": 0,
            "max": 20,
        },
    },
    "categorical_options": {
        "sexo": ["M", "F"],
        "cor_raca": ["Branca", "Preta", "Parda", "Amarela", "Indigena"],
        "setor": ["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"],
        "regiao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
    },
}


def get_api_url() -> str:
    env_api_url = os.getenv("API_URL")
    app_env = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT")

    try:
        secrets_api_url = st.secrets.get("API_URL")
        secrets_app_env = st.secrets.get("APP_ENV") or st.secrets.get("ENVIRONMENT")
    except Exception:
        secrets_api_url = None
        secrets_app_env = None

    if env_api_url:
        return env_api_url.rstrip("/")

    if secrets_api_url:
        return str(secrets_api_url).rstrip("/")

    if app_env and app_env.lower() in {"production", "prod"}:
        return PRODUCTION_API_URL.rstrip("/")

    if secrets_app_env and str(secrets_app_env).lower() in {"production", "prod"}:
        return PRODUCTION_API_URL.rstrip("/")

    if (
        os.getenv("RENDER")
        or os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("STREAMLIT_SERVER_HEADLESS")
        or os.getenv("STREAMLIT_RUNTIME")
    ):
        return PRODUCTION_API_URL.rstrip("/")

    return LOCAL_API_URL.rstrip("/")


st.set_page_config(
    page_title="Rendimento Predictor",
    page_icon="📊",
    layout="centered",
)


API_URL = get_api_url()


def load_json(endpoint: str) -> dict:
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def load_metadata() -> dict:
    try:
        return load_json("/metadata")
    except requests.exceptions.RequestException:
        return FALLBACK_METADATA


def get_numeric_constraint(metadata: dict, field: str) -> tuple[int, int]:
    fallback = FALLBACK_METADATA["numeric_constraints"][field]
    constraints = metadata.get("numeric_constraints", {}).get(field, {})

    return (
        int(constraints.get("min", fallback["min"])),
        int(constraints.get("max", fallback["max"])),
    )


def get_categorical_options(metadata: dict, field: str) -> list[str]:
    fallback = FALLBACK_METADATA["categorical_options"][field]
    options = metadata.get("categorical_options", {}).get(field)

    if not options:
        return fallback

    return list(options)


METADATA = load_metadata()
IDADE_MIN, IDADE_MAX = get_numeric_constraint(METADATA, "idade")
ANOS_ESTUDO_MIN, ANOS_ESTUDO_MAX = get_numeric_constraint(METADATA, "anos_estudo")
SEXO_OPTIONS = get_categorical_options(METADATA, "sexo")
COR_RACA_OPTIONS = get_categorical_options(METADATA, "cor_raca")
SETOR_OPTIONS = get_categorical_options(METADATA, "setor")
REGIAO_OPTIONS = get_categorical_options(METADATA, "regiao")


def render_model_metrics(model_info: dict) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RMSE", model_info["rmse"])

    with col2:
        st.metric("MAE", model_info["mae"])

    with col3:
        st.metric("R²", model_info["r2"])

    st.json(model_info)


def render_feature_importance(feature_importance_data: dict) -> None:
    st.write(f"Modelo: `{feature_importance_data['model_name']}`")
    st.write(f"Tipo: `{feature_importance_data['model_type']}`")

    feature_importance = feature_importance_data["feature_importance"]

    if not feature_importance:
        st.warning("Nenhuma importância de variável encontrada.")
        return

    st.bar_chart(
        {
            item["feature"]: item["importance"]
            for item in feature_importance
        }
    )

    st.dataframe(feature_importance, use_container_width=True)


def render_model_comparison(comparison_data: dict) -> None:
    st.write(f"Fonte dos dados: `{comparison_data['data_source']}`")
    st.write(f"Melhor modelo por RMSE: `{comparison_data['best_model_by_rmse']}`")
    st.write(f"Número de linhas: `{comparison_data['n_rows']}`")

    models = comparison_data["models"]

    comparison_table = [
        {
            "modelo": model["model_name"],
            "rmse": model["rmse"],
            "mae": model["mae"],
            "r2": model["r2"],
            "cv_r2_mean": model["cv_r2_mean"],
            "cv_r2_std": model["cv_r2_std"],
        }
        for model in models
    ]

    st.dataframe(comparison_table, use_container_width=True)

    st.subheader("RMSE por modelo")

    st.bar_chart(
        {
            model["model_name"]: model["rmse"]
            for model in models
        }
    )

    st.subheader("R² por modelo")

    st.bar_chart(
        {
            model["model_name"]: model["r2"]
            for model in models
        }
    )


st.title("📊 Rendimento Predictor API")

st.write(
    """
    Aplicação simples para consumir a API de predição de rendimento por hora.

    O projeto combina Machine Learning, FastAPI, Streamlit, testes automatizados,
    CI/CD e deploy. O modelo atual em produção ainda é um baseline treinado com
    dados sintéticos, enquanto o pipeline de dados reais já está sendo preparado
    separadamente.
    """
)


with st.sidebar:
    st.header("Informações")
    st.write("API pública:")
    st.code(API_URL)

    st.write("Endpoints principais:")
    st.markdown("- `/health`")
    st.markdown("- `/features`")
    st.markdown("- `/model-info`")
    st.markdown("- `/model-info/real`")
    st.markdown("- `/model-info/production`")
    st.markdown("- `/model-comparison`")
    st.markdown("- `/feature-importance`")
    st.markdown("- `/feature-importance/real`")
    st.markdown("- `/predict`")
    st.markdown("- `/history`")

    st.warning(
        "O modelo em produção ainda usa dados sintéticos. "
        "O pipeline de dados reais está em desenvolvimento."
    )


tab_predict, tab_models, tab_comparison, tab_history, tab_about = st.tabs(
    [
        "Predição",
        "Modelos",
        "Comparação",
        "Histórico",
        "Sobre",
    ]
)


with tab_predict:
    st.subheader("Preencha os dados para predição")

    idade = st.slider(
        "Idade",
        min_value=IDADE_MIN,
        max_value=IDADE_MAX,
        value=35,
    )

    sexo = st.selectbox(
        "Sexo",
        options=SEXO_OPTIONS,
    )

    cor_raca = st.selectbox(
        "Cor/Raça",
        options=COR_RACA_OPTIONS,
    )

    anos_estudo = st.slider(
        "Anos de estudo",
        min_value=ANOS_ESTUDO_MIN,
        max_value=ANOS_ESTUDO_MAX,
        value=12,
    )

    setor = st.selectbox(
        "Setor",
        options=SETOR_OPTIONS,
    )

    regiao = st.selectbox(
        "Região",
        options=REGIAO_OPTIONS,
    )

    payload = {
        "idade": idade,
        "sexo": sexo,
        "cor_raca": cor_raca,
        "anos_estudo": anos_estudo,
        "setor": setor,
        "regiao": regiao,
    }

    if st.button("Prever rendimento"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            st.success("Predição realizada com sucesso!")

            st.metric(
                label="Rendimento por hora previsto",
                value=f"R$ {result['rendimento_hora_previsto']:.2f}",
            )

            intervalo = result["intervalo_confianca"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Limite inferior",
                    value=f"R$ {intervalo['min']:.2f}",
                )

            with col2:
                st.metric(
                    label="Limite superior",
                    value=f"R$ {intervalo['max']:.2f}",
                )

            st.subheader("Dados enviados")
            st.json(result["features_usadas"])

            st.subheader("Modelo utilizado")
            st.write(result["modelo"])

        except requests.exceptions.RequestException as error:
            st.error("Erro ao chamar a API.")
            st.exception(error)


with tab_models:
    st.subheader("Métricas e importância das variáveis")

    st.write(
        """
        O modelo legado é o baseline sintético antigo. O modelo candidato representa
        o pipeline real anterior. O modelo de produção é o XGBoost treinado com
        microdados reais da PNAD.
        """
    )

    model_option = st.radio(
        "Escolha o modelo",
        options=[
            "Modelo legado",
            "Modelo candidato",
            "Modelo de produção",
        ],
        horizontal=True,
    )

    if model_option == "Modelo legado":
        model_info_endpoint = "/model-info"
        feature_importance_endpoint = "/feature-importance"
    elif model_option == "Modelo candidato":
        model_info_endpoint = "/model-info/real"
        feature_importance_endpoint = "/feature-importance/real"
    else:
        model_info_endpoint = "/model-info/production"
        feature_importance_endpoint = "/feature-importance/production"

    col_metrics, col_importance = st.columns(2)

    with col_metrics:
        st.markdown("### Métricas")

        if st.button("Carregar métricas"):
            try:
                model_info = load_json(model_info_endpoint)
                render_model_metrics(model_info)

            except requests.exceptions.RequestException as error:
                st.error("Erro ao carregar métricas do modelo.")
                st.exception(error)

    with col_importance:
        st.markdown("### Feature importance")

        if st.button("Carregar importância das variáveis"):
            try:
                feature_importance = load_json(feature_importance_endpoint)
                render_feature_importance(feature_importance)

            except requests.exceptions.RequestException as error:
                st.error("Erro ao carregar importância das variáveis.")
                st.exception(error)


with tab_comparison:
    st.subheader("Comparação de modelos")

    st.write(
        """
        Esta seção compara diferentes modelos treinados sobre o dataset padronizado
        do pipeline real:

        - Regressão Linear
        - Random Forest
        - XGBoost

        A escolha do melhor modelo é feita pelo menor RMSE.
        """
    )

    if st.button("Carregar comparação de modelos"):
        try:
            comparison = load_json("/model-comparison")
            render_model_comparison(comparison)

        except requests.exceptions.RequestException as error:
            st.error("Erro ao carregar comparação de modelos.")
            st.exception(error)


with tab_history:
    st.subheader("Histórico de predições")

    history_limit = st.slider(
        "Limite de registros",
        min_value=1,
        max_value=50,
        value=10,
    )

    if st.button("Carregar histórico"):
        try:
            history = load_json(f"/history?limit={history_limit}")
            total_returned = history["total_returned"]
            predictions = history["predictions"]

            st.metric("Total retornado", total_returned)

            if not predictions:
                st.warning("Nenhuma predição encontrada no histórico.")
            else:
                st.dataframe(predictions, use_container_width=True)

        except requests.exceptions.RequestException as error:
            st.error("Erro ao carregar histórico de predições.")
            st.exception(error)


with tab_about:
    st.subheader("Sobre o projeto")

    st.write(
        """
        Este projeto foi construído para demonstrar um fluxo completo de aplicação
        preditiva:

        - geração e preparação de dados;
        - análise exploratória;
        - treinamento de modelo;
        - comparação entre modelos;
        - API com FastAPI;
        - testes automatizados;
        - CI/CD com GitHub Actions;
        - Docker;
        - deploy da API no Render;
        - frontend no Streamlit Cloud;
        - separação entre modelo em produção e modelo candidato;
        - análise de importância das variáveis.
        """
    )

    st.info(
        "A próxima grande evolução do projeto é substituir o arquivo padronizado "
        "de exemplo por dados reais completos da PNAD/IBGE."
    )

    st.write("Links principais:")

    st.markdown(
        "- API: https://rendimento-predictor-api.onrender.com/"
    )
    st.markdown(
        "- Swagger: https://rendimento-predictor-api.onrender.com/docs"
    )
    st.markdown(
        "- Frontend: https://rendimento-predictor-api.streamlit.app/"
    )
