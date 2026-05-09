import argparse
from pathlib import Path

import pandas as pd


REPORT_PATH = Path("data/processed/pnad_inspection_report.txt")


READ_ATTEMPTS = [
    {"sep": ","},
    {"sep": ";"},
    {"sep": ",", "encoding": "latin1"},
    {"sep": ";", "encoding": "latin1"},
]


def read_pnad_file(file_path: Path) -> pd.DataFrame:
    errors = []

    for options in READ_ATTEMPTS:
        try:
            return pd.read_csv(file_path, **options)
        except Exception as exc:
            errors.append(f"{options}: {exc}")

    formatted_errors = "\n".join(f"- {error}" for error in errors)
    raise ValueError(
        "Nao foi possivel ler o arquivo PNAD com as tentativas iniciais.\n"
        f"Tentativas realizadas:\n{formatted_errors}"
    )


def build_report(file_path: Path, df: pd.DataFrame) -> str:
    report_sections = [
        "Inspecao exploratoria PNAD",
        "=" * 28,
        "",
        "Caminho do arquivo:",
        str(file_path),
        "",
        "Shape:",
        str(df.shape),
        "",
        "Primeiras 5 linhas:",
        df.head().to_string(),
        "",
        "Colunas:",
        str(list(df.columns)),
        "",
        "Dtypes:",
        df.dtypes.to_string(),
        "",
        "Valores ausentes por coluna:",
        df.isna().sum().to_string(),
        "",
        "Estatisticas descritivas:",
        df.describe(include="all").to_string(),
        "",
    ]

    return "\n".join(report_sections)


def inspect_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    df = read_pnad_file(file_path)
    report = build_report(file_path, df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspeciona um arquivo bruto da PNAD Continua."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Caminho do arquivo PNAD a ser inspecionado.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        report = inspect_file(args.file_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(report)
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
