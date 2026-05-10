import pytest

from scripts.download_pnad import (
    build_output_path,
    download_pnad_file,
    extract_filename,
)


def test_extract_filename_from_url():
    url = "https://example.com/pnad/arquivo%20pnad.zip?download=1"

    assert extract_filename(url) == "arquivo pnad.zip"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "ftp://example.com/arquivo.zip",
        "https://example.com/",
    ],
)
def test_extract_filename_rejects_invalid_url(url):
    with pytest.raises(ValueError):
        extract_filename(url)


def test_build_output_path_uses_raw_pnad_directory(tmp_path):
    output_path = build_output_path(
        "https://example.com/files/pnad.zip",
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "pnad.zip"


def test_download_pnad_file_skips_existing_file(tmp_path):
    existing_file = tmp_path / "pnad.zip"
    existing_file.write_bytes(b"existing data")

    result = download_pnad_file(
        "https://example.com/files/pnad.zip",
        output_dir=tmp_path,
    )

    assert result.output_path == existing_file
    assert result.size_bytes == len(b"existing data")
    assert result.downloaded is False
    assert existing_file.read_bytes() == b"existing data"
