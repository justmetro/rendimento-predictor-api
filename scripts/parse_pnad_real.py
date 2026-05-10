import argparse
from pathlib import Path

import pandas as pd


PROCESSED_DATA_PATH = Path("data/processed/pnad_real_processed.csv")

PNAD_COLSPECS = [
    (5, 7),
    (94, 95),
    (103, 106),
    (106, 107),
    (507, 509),
    (519, 521),
    (544, 552),
    (552, 560),
]

PNAD_COLUMN_NAMES = [
    "UF",
    "V2007",
    "V2009",
    "V2010",
    "VD3005",
    "VD4010",
    "VD4019",
    "VD4020",
]

# Posicoes SAS sao 1-based; pandas.read_fwf usa inicio 0-based e fim exclusivo.
FINAL_COLUMNS = [
    "idade",
    "sexo",
    "cor_raca",
    "anos_estudo",
    "setor",
    "regiao",
    "rendimento_hora",
]

SEXO_MAP = {
    "1": "M",
    "2": "F",
}

COR_RACA_MAP = {
    "1": "Branca",
    "2": "Preta",
    "3": "Amarela",
    "4": "Parda",
    "5": "Indigena",
}

REGIAO_BY_UF = {
    "11": "Norte",
    "12": "Norte",
    "13": "Norte",
    "14": "Norte",
    "15": "Norte",
    "16": "Norte",
    "17": "Norte",
    "21": "Nordeste",
    "22": "Nordeste",
    "23": "Nordeste",
    "24": "Nordeste",
    "25": "Nordeste",
    "26": "Nordeste",
    "27": "Nordeste",
    "28": "Nordeste",
    "29": "Nordeste",
    "31": "Sudeste",
    "32": "Sudeste",
    "33": "Sudeste",
    "35": "Sudeste",
    "41": "Sul",
    "42": "Sul",
    "43": "Sul",
    "50": "Centro-Oeste",
    "51": "Centro-Oeste",
    "52": "Centro-Oeste",
    "53": "Centro-Oeste",
}


def read_pnad_fixed_width(file_path: Path, sample_size: int | None = None) -> pd.DataFrame:
    return pd.read_fwf(
        file_path,
        colspecs=PNAD_COLSPECS,
        names=PNAD_COLUMN_NAMES,
        dtype=str,
        header=None,
        nrows=sample_size,
    )


def map_setor(value: object) -> object:
    code = pd.to_numeric(value, errors="coerce")

    if pd.isna(code):
        return pd.NA

    code = int(code)

    if code in {1, 2}:
        return "Agricultura"
    if code in {3, 4}:
        return "Industria"
    if code == 5:
        return "Construcao"
    if code in {6, 7}:
        return "Comercio"

    return "Servicos"


def transform_pnad_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()

    df["idade"] = pd.to_numeric(raw_df["V2009"], errors="coerce")
    df["sexo"] = raw_df["V2007"].map(SEXO_MAP)
    df["cor_raca"] = raw_df["V2010"].map(COR_RACA_MAP)
    df["anos_estudo"] = pd.to_numeric(raw_df["VD3005"], errors="coerce")
    df["setor"] = raw_df["VD4010"].apply(map_setor)
    df["regiao"] = raw_df["UF"].map(REGIAO_BY_UF)

    rendimento_habitual = pd.to_numeric(raw_df["VD4019"], errors="coerce")
    rendimento_efetivo = pd.to_numeric(raw_df["VD4020"], errors="coerce")
    rendimento_mensal = rendimento_habitual.where(
        rendimento_habitual.notna() & (rendimento_habitual > 0),
        rendimento_efetivo,
    )

    # Aproximacao inicial: usa 160 horas mensais fixas ate incorporarmos horas trabalhadas da PNAD.
    df["rendimento_hora"] = rendimento_mensal / 160

    return df[FINAL_COLUMNS]


def clean_pnad_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=FINAL_COLUMNS)

    df = df[
        (df["idade"] >= 18)
        & (df["idade"] <= 80)
        & (df["anos_estudo"] >= 0)
        & (df["anos_estudo"] <= 25)
        & (df["rendimento_hora"] > 0)
    ]

    if df.empty:
        return df.reset_index(drop=True)

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


def parse_pnad_file(
    file_path: Path,
    output_path: Path = PROCESSED_DATA_PATH,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, int]:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    raw_df = read_pnad_fixed_width(file_path, sample_size=sample_size)
    processed_df = clean_pnad_data(transform_pnad_data(raw_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False, encoding="utf-8")

    return processed_df, len(raw_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parseia microdados PNAD Continua de largura fixa."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Caminho do arquivo bruto PNAD de largura fixa.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Numero maximo de linhas a ler para testes rapidos.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        processed_df, rows_read = parse_pnad_file(
            args.file_path,
            sample_size=args.sample_size,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Linhas lidas: {rows_read}")
    print(f"Linhas finais: {processed_df.shape[0]}")
    print(f"Colunas finais: {list(processed_df.columns)}")
    print(processed_df.head())
    print(f"Caminho salvo: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
