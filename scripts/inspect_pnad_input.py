import argparse
import re
from pathlib import Path


REPORT_PATH = Path("data/processed/pnad_input_inspection_report.txt")

ENCODINGS = ("utf-8", "latin1", "cp1252")

IMPORTANT_TERMS = (
    "V2009",
    "V2007",
    "V2010",
    "VD3004",
    "VD3005",
    "VD4010",
    "VD4019",
    "VD4020",
    "UF",
    "UPA",
    "Estrato",
    "V1028",
)

POSITION_LINE_PATTERNS = (
    re.compile(
        r"\b(posicao|posição|inicio|início|fim|largura|tamanho|coluna|byte)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*[A-Z]{1,3}\d{0,4}\b.*\b\d+\b.*\b\d+\b", re.IGNORECASE),
)


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    errors = []

    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    formatted_errors = "\n".join(f"- {error}" for error in errors)
    raise ValueError(
        "Nao foi possivel ler o arquivo input/layout da PNAD.\n"
        f"Tentativas realizadas:\n{formatted_errors}"
    )


def find_relevant_lines(lines: list[str], terms: tuple[str, ...]) -> list[str]:
    normalized_terms = tuple(term.casefold() for term in terms)
    relevant_lines = []

    for line_number, line in enumerate(lines, start=1):
        normalized_line = line.casefold()
        if any(term in normalized_line for term in normalized_terms):
            relevant_lines.append(f"{line_number}: {line}")

    return relevant_lines


def find_position_width_lines(lines: list[str]) -> list[str]:
    position_lines = []

    for line_number, line in enumerate(lines, start=1):
        if any(pattern.search(line) for pattern in POSITION_LINE_PATTERNS):
            position_lines.append(f"{line_number}: {line}")

    return position_lines


def build_report(
    file_path: Path,
    encoding: str,
    lines: list[str],
    relevant_lines: list[str],
    position_width_lines: list[str],
) -> str:
    first_lines = lines[:80]

    report_sections = [
        "Inspecao do input/layout PNAD",
        "=" * 30,
        "",
        "Caminho do arquivo:",
        str(file_path),
        "",
        "Encoding usado:",
        encoding,
        "",
        "Quantidade total de linhas:",
        str(len(lines)),
        "",
        "Primeiras 80 linhas:",
        *[f"{index}: {line}" for index, line in enumerate(first_lines, start=1)],
        "",
        "Linhas com termos importantes:",
        *(relevant_lines or ["Nenhuma linha encontrada."]),
        "",
        "Linhas que parecem definir posicao/largura:",
        *(position_width_lines or ["Nenhuma linha encontrada."]),
        "",
    ]

    return "\n".join(report_sections)


def inspect_input_file(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    text, encoding = read_text_with_fallback(file_path)
    lines = text.splitlines()
    relevant_lines = find_relevant_lines(lines, IMPORTANT_TERMS)
    position_width_lines = find_position_width_lines(lines)
    report = build_report(
        file_path=file_path,
        encoding=encoding,
        lines=lines,
        relevant_lines=relevant_lines,
        position_width_lines=position_width_lines,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspeciona o arquivo input/layout da PNAD Continua."
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Caminho do arquivo input/layout da PNAD.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        report = inspect_input_file(args.file_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(report)
    print(f"Relatorio salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    main()
