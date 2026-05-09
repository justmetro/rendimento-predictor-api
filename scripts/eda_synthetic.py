import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "data/processed/synthetic_rendimento.csv"
REPORT_PATH = "data/processed/eda_summary.txt"
PLOTS_DIR = "data/processed/plots"


def save_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["rendimento_hora"], bins=30)
    plt.title("Distribuição do rendimento por hora")
    plt.xlabel("Rendimento por hora")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/target_distribution.png")
    plt.close()


def save_rendimento_by_regiao(df: pd.DataFrame) -> None:
    media_por_regiao = (
        df.groupby("regiao")["rendimento_hora"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    media_por_regiao.plot(kind="bar")
    plt.title("Média de rendimento por região")
    plt.xlabel("Região")
    plt.ylabel("Rendimento médio por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/rendimento_by_regiao.png")
    plt.close()


def save_rendimento_by_setor(df: pd.DataFrame) -> None:
    media_por_setor = (
        df.groupby("setor")["rendimento_hora"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    media_por_setor.plot(kind="bar")
    plt.title("Média de rendimento por setor")
    plt.xlabel("Setor")
    plt.ylabel("Rendimento médio por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/rendimento_by_setor.png")
    plt.close()


def save_idade_vs_rendimento(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["idade"], df["rendimento_hora"], alpha=0.5)
    plt.title("Idade vs rendimento por hora")
    plt.xlabel("Idade")
    plt.ylabel("Rendimento por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/idade_vs_rendimento.png")
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. "
            "Rode primeiro: python -m scripts.generate_data"
        )

    df = pd.read_csv(DATA_PATH)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("EDA - Dataset Sintético de Rendimento\n")
        file.write("=" * 45 + "\n\n")

        file.write("Shape do dataset:\n")
        file.write(f"{df.shape}\n\n")

        file.write("Colunas:\n")
        file.write(f"{list(df.columns)}\n\n")

        file.write("Tipos das variáveis:\n")
        file.write(f"{df.dtypes}\n\n")

        file.write("Valores ausentes:\n")
        file.write(f"{df.isna().sum()}\n\n")

        file.write("Estatísticas descritivas:\n")
        file.write(f"{df.describe()}\n\n")

        file.write("Distribuição por sexo:\n")
        file.write(f"{df['sexo'].value_counts()}\n\n")

        file.write("Distribuição por região:\n")
        file.write(f"{df['regiao'].value_counts()}\n\n")

        file.write("Média de rendimento por região:\n")
        file.write(
            f"{df.groupby('regiao')['rendimento_hora'].mean().sort_values(ascending=False)}\n\n"
        )

        file.write("Média de rendimento por setor:\n")
        file.write(
            f"{df.groupby('setor')['rendimento_hora'].mean().sort_values(ascending=False)}\n\n"
        )

    save_target_distribution(df)
    save_rendimento_by_regiao(df)
    save_rendimento_by_setor(df)
    save_idade_vs_rendimento(df)

    print(f"Relatório de EDA salvo em: {REPORT_PATH}")
    print(f"Gráficos salvos em: {PLOTS_DIR}")


if __name__ == "__main__":
    main()