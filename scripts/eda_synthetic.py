import os

import pandas as pd


DATA_PATH = "data/processed/synthetic_rendimento.csv"
REPORT_PATH = "data/processed/eda_summary.txt"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. "
            "Rode primeiro: python -m scripts.generate_data"
        )

    df = pd.read_csv(DATA_PATH)

    os.makedirs("data/processed", exist_ok=True)

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

    print(f"Relatório de EDA salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()