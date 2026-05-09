import os

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "data/processed/pnad_real_processed.csv"
REPORT_PATH = "data/processed/real_eda_summary.txt"
PLOTS_DIR = "data/processed/real_plots"


def save_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["rendimento_hora"], bins=20)
    plt.title("Distribuição do rendimento por hora - Dados reais")
    plt.xlabel("Rendimento por hora")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/real_target_distribution.png")
    plt.close()


def save_rendimento_by_regiao(df: pd.DataFrame) -> None:
    media_por_regiao = (
        df.groupby("regiao")["rendimento_hora"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    media_por_regiao.plot(kind="bar")
    plt.title("Média de rendimento por região - Dados reais")
    plt.xlabel("Região")
    plt.ylabel("Rendimento médio por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/real_rendimento_by_regiao.png")
    plt.close()


def save_rendimento_by_setor(df: pd.DataFrame) -> None:
    media_por_setor = (
        df.groupby("setor")["rendimento_hora"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    media_por_setor.plot(kind="bar")
    plt.title("Média de rendimento por setor - Dados reais")
    plt.xlabel("Setor")
    plt.ylabel("Rendimento médio por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/real_rendimento_by_setor.png")
    plt.close()


def save_anos_estudo_vs_rendimento(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["anos_estudo"], df["rendimento_hora"], alpha=0.6)
    plt.title("Anos de estudo vs rendimento por hora - Dados reais")
    plt.xlabel("Anos de estudo")
    plt.ylabel("Rendimento por hora")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/real_anos_estudo_vs_rendimento.png")
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. "
            "Rode primeiro: python -m scripts.prepare_real_data"
        )

    df = pd.read_csv(DATA_PATH)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("EDA - Dados Reais Padronizados\n")
        file.write("=" * 35 + "\n\n")

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

        file.write("Distribuição por cor/raça:\n")
        file.write(f"{df['cor_raca'].value_counts()}\n\n")

        file.write("Distribuição por região:\n")
        file.write(f"{df['regiao'].value_counts()}\n\n")

        file.write("Distribuição por setor:\n")
        file.write(f"{df['setor'].value_counts()}\n\n")

        file.write("Média de rendimento por sexo:\n")
        file.write(
            f"{df.groupby('sexo')['rendimento_hora'].mean().sort_values(ascending=False)}\n\n"
        )

        file.write("Média de rendimento por cor/raça:\n")
        file.write(
            f"{df.groupby('cor_raca')['rendimento_hora'].mean().sort_values(ascending=False)}\n\n"
        )

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
    save_anos_estudo_vs_rendimento(df)

    print(f"Relatório de EDA real salvo em: {REPORT_PATH}")
    print(f"Gráficos reais salvos em: {PLOTS_DIR}")


if __name__ == "__main__":
    main()