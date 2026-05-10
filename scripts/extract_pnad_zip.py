import argparse
import zipfile
from pathlib import Path


EXTRACTED_PNAD_DIR = Path("data/raw/pnad/extracted")


class ExtractResult:
    def __init__(
        self,
        zip_path: Path,
        output_dir: Path,
        extracted_files: list[Path],
        extracted: bool,
    ) -> None:
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.extracted_files = extracted_files
        self.extracted = extracted


def build_extract_dir(
    zip_path: Path,
    extracted_base_dir: Path = EXTRACTED_PNAD_DIR,
) -> Path:
    return extracted_base_dir / zip_path.stem


def list_extracted_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []

    return sorted(path for path in output_dir.rglob("*") if path.is_file())


def remove_directory(output_dir: Path) -> None:
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()

    output_dir.rmdir()


def extract_pnad_zip(
    zip_path: Path,
    extracted_base_dir: Path = EXTRACTED_PNAD_DIR,
    force: bool = False,
) -> ExtractResult:
    if not zip_path.exists():
        raise FileNotFoundError(f"Arquivo ZIP nao encontrado: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Arquivo nao e um ZIP valido: {zip_path}")

    output_dir = build_extract_dir(zip_path, extracted_base_dir)

    if output_dir.exists() and not force:
        return ExtractResult(
            zip_path=zip_path,
            output_dir=output_dir,
            extracted_files=list_extracted_files(output_dir),
            extracted=False,
        )

    if output_dir.exists() and force:
        remove_directory(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
    except (zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(f"Falha ao extrair arquivo PNAD: {exc}") from exc

    return ExtractResult(
        zip_path=zip_path,
        output_dir=output_dir,
        extracted_files=list_extracted_files(output_dir),
        extracted=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai um arquivo ZIP bruto da PNAD/IBGE."
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Caminho do arquivo ZIP da PNAD a extrair.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove a pasta extraida existente e extrai novamente.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = extract_pnad_zip(args.zip_path, force=args.force)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"ZIP: {result.zip_path}")
    print(f"Pasta de saida: {result.output_dir}")

    if result.extracted:
        print("Extracao concluida com sucesso.")
    else:
        print("Pasta de destino ja existe. Use --force para extrair novamente.")

    print("Arquivos extraidos:")
    for extracted_file in result.extracted_files:
        print(f"- {extracted_file}")


if __name__ == "__main__":
    main()
