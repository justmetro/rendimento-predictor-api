import os
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/pnad_real.csv")
PROCESSED_DATA_PATH = Path("data/processed/pnad_real_processed.csv")


REQUIRED_COLUMNS = [
    "idade",
    "sexo",
    "cor_raca",
    "anos_estudo",
    "setor",
    "regiao",
    "rendimento_hora",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "O arquivo de dados reais não contém todas as colunas esperadas. "
            f"Colunas ausentes: {missing_columns}. "
            f"Colunas obrigatórias: {REQUIRED_COLUMNS}"
        )


def clean_real_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df)

    df = df[REQUIRED_COLUMNS].copy()

    df = df.drop_duplicates()

    df = df.dropna(
        subset=[
            "idade",
            "sexo",
            "cor_raca",
            "anos_estudo",
            "setor",
            "regiao",
            "rendimento_hora",
        ]
    )

    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    df["anos_estudo"] = pd.to_numeric(df["anos_estudo"], errors="coerce")
    df["rendimento_hora"] = pd.to_numeric(df["rendimento_hora"], errors="coerce")

    df = df.dropna(
        subset=[
            "idade",
            "anos_estudo",
            "rendimento_hora",
        ]
    )

    df = df[
        (df["idade"] >= 18)
        & (df["idade"] <= 80)
        & (df["anos_estudo"] >= 0)
        & (df["anos_estudo"] <= 25)
        & (df["rendimento_hora"] > 0)
    ]

    q1 = df["rendimento_hora"].quantile(0.25)
    q3 = df["rendimento_hora"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[
        (df["rendimento_hora"] >= lower_bound)
        & (df["rendimento_hora"] <= upper_bound)
    ]

    df["idade"] = df["idade"].astype(int)
    df["anos_estudo"] = df["anos_estudo"].astype(int)

    return df.reset_index(drop=True)


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_DATA_PATH}\n\n"
            "Crie um arquivo CSV em data/raw/pnad_real.csv com as colunas:\n"
            f"{REQUIRED_COLUMNS}\n\n"
            "Por enquanto, este script espera os dados já padronizados com esses nomes."
        )

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    processed_df = clean_real_data(df)

    processed_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")

    print(f"Arquivo processado salvo em: {PROCESSED_DATA_PATH}")
    print(f"Linhas finais: {processed_df.shape[0]}")
    print(f"Colunas finais: {processed_df.shape[1]}")
    print(processed_df.head())


if __name__ == "__main__":
    main()