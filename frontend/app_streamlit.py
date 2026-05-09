import requests
import streamlit as st


API_URL = "https://rendimento-predictor-api.onrender.com"


st.set_page_config(
    page_title="Rendimento Predictor",
    page_icon="📊",
    layout="centered",
)


def load_json(endpoint: str) -> dict:
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


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
    st.markdown("- `/feature-importance`")
    st.markdown("- `/feature-importance/real`")
    st.markdown("- `/predict`")

    st.warning(
        "O modelo em produção ainda usa dados sintéticos. "
        "O pipeline de dados reais está em desenvolvimento."
    )


tab_predict, tab_models, tab_about = st.tabs(
    [
        "Predição",
        "Modelos",
        "Sobre",
    ]
)


with tab_predict:
    st.subheader("Preencha os dados para predição")

    idade = st.slider(
        "Idade",
        min_value=18,
        max_value=80,
        value=35,
    )

    sexo = st.selectbox(
        "Sexo",
        options=["M", "F"],
    )

    cor_raca = st.selectbox(
        "Cor/Raça",
        options=["Branca", "Preta", "Parda", "Amarela", "Indigena"],
    )

    anos_estudo = st.slider(
        "Anos de estudo",
        min_value=0,
        max_value=20,
        value=12,
    )

    setor = st.selectbox(
        "Setor",
        options=["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"],
    )

    regiao = st.selectbox(
        "Região",
        options=["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
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
        O modelo em produção é o modelo atualmente utilizado pelo endpoint `/predict`.
        O modelo candidato representa o pipeline separado para dados reais padronizados.
        """
    )

    model_option = st.radio(
        "Escolha o modelo",
        options=[
            "Modelo em produção",
            "Modelo candidato",
        ],
        horizontal=True,
    )

    if model_option == "Modelo em produção":
        model_info_endpoint = "/model-info"
        feature_importance_endpoint = "/feature-importance"
    else:
        model_info_endpoint = "/model-info/real"
        feature_importance_endpoint = "/feature-importance/real"

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


with tab_about:
    st.subheader("Sobre o projeto")

    st.write(
        """
        Este projeto foi construído para demonstrar um fluxo completo de aplicação
        preditiva:

        - geração e preparação de dados;
        - análise exploratória;
        - treinamento de modelo;
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