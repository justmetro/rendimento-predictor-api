import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


RAW_PNAD_DIR = Path("data/raw/pnad")


@dataclass(frozen=True)
class DownloadResult:
    url: str
    output_path: Path
    size_bytes: int | None
    downloaded: bool


def extract_filename(url: str) -> str:
    parsed_url = urlparse(url.strip())

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("URL invalida. Informe uma URL http(s) completa.")

    filename = Path(unquote(parsed_url.path)).name
    if not filename:
        raise ValueError("URL invalida. Nao foi possivel identificar o nome do arquivo.")

    return filename


def build_output_path(url: str, output_dir: Path = RAW_PNAD_DIR) -> Path:
    return output_dir / extract_filename(url)


def download_pnad_file(
    url: str,
    output_dir: Path = RAW_PNAD_DIR,
    force: bool = False,
) -> DownloadResult:
    url = url.strip()
    output_path = build_output_path(url, output_dir)

    if output_path.exists() and not force:
        return DownloadResult(
            url=url,
            output_path=output_path,
            size_bytes=output_path.stat().st_size,
            downloaded=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.part")

    try:
        with urlopen(url, timeout=60) as response, temp_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)

        temp_path.replace(output_path)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Falha ao baixar arquivo PNAD: {exc}") from exc

    return DownloadResult(
        url=url,
        output_path=output_path,
        size_bytes=output_path.stat().st_size if output_path.exists() else None,
        downloaded=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa um arquivo bruto da PNAD/IBGE para data/raw/pnad."
    )
    parser.add_argument("url", help="URL http(s) do arquivo PNAD a baixar.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve o arquivo local caso ele ja exista.",
    )

    return parser.parse_args()


def format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "desconhecido"

    return f"{size_bytes} bytes"


def main() -> None:
    args = parse_args()

    try:
        result = download_pnad_file(args.url, force=args.force)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"URL: {result.url}")
    print(f"Caminho de saida: {result.output_path}")
    print(f"Tamanho baixado: {format_size(result.size_bytes)}")

    if result.downloaded:
        print("Download concluido com sucesso.")
    else:
        print("Arquivo ja existe. Use --force para baixar novamente.")


if __name__ == "__main__":
    main()
