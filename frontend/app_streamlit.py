import requests
import streamlit as st


API_URL = "https://rendimento-predictor-api.onrender.com"


st.set_page_config(
    page_title="Rendimento Predictor",
    page_icon="📊",
    layout="centered",
)


st.title("📊 Rendimento Predictor API")

st.write(
    """
    Aplicação simples para consumir a API de predição de rendimento por hora.
    
    O modelo atual é um baseline treinado com dados sintéticos, usado para validar
    o fluxo completo de Machine Learning, API, testes, CI/CD e deploy.
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
    st.markdown("- `/predict`")


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

        st.subheader("Modelo")
        st.write(result["modelo"])

    except requests.exceptions.RequestException as error:
        st.error("Erro ao chamar a API.")
        st.exception(error)


st.divider()

st.subheader("Informações do modelo")

if st.button("Carregar métricas do modelo"):
    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=30,
        )

        response.raise_for_status()

        model_info = response.json()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("RMSE", model_info["rmse"])

        with col2:
            st.metric("MAE", model_info["mae"])

        with col3:
            st.metric("R²", model_info["r2"])

        st.json(model_info)

    except requests.exceptions.RequestException as error:
        st.error("Erro ao carregar informações do modelo.")
        st.exception(error)