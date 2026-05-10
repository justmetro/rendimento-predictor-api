import zipfile

import pytest

from scripts.extract_pnad_zip import build_extract_dir, extract_pnad_zip


def create_test_zip(zip_path):
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("dados.txt", "conteudo")
        archive.writestr("subdir/metadados.txt", "metadata")


def test_build_extract_dir_uses_zip_stem(tmp_path):
    zip_path = tmp_path / "PNADC_2023_trimestre1.zip"

    assert build_extract_dir(zip_path, tmp_path / "extracted") == (
        tmp_path / "extracted" / "PNADC_2023_trimestre1"
    )


def test_extract_pnad_zip_extracts_small_zip(tmp_path):
    zip_path = tmp_path / "PNADC_2023_trimestre1.zip"
    create_test_zip(zip_path)

    result = extract_pnad_zip(zip_path, extracted_base_dir=tmp_path / "extracted")

    assert result.extracted is True
    assert result.output_dir == tmp_path / "extracted" / "PNADC_2023_trimestre1"
    assert (result.output_dir / "dados.txt").read_text() == "conteudo"
    assert (result.output_dir / "subdir" / "metadados.txt").read_text() == "metadata"
    assert result.extracted_files == [
        result.output_dir / "dados.txt",
        result.output_dir / "subdir" / "metadados.txt",
    ]


def test_extract_pnad_zip_skips_existing_output_dir(tmp_path):
    zip_path = tmp_path / "PNADC_2023_trimestre1.zip"
    create_test_zip(zip_path)
    output_dir = tmp_path / "extracted" / "PNADC_2023_trimestre1"
    output_dir.mkdir(parents=True)
    existing_file = output_dir / "dados.txt"
    existing_file.write_text("existente")

    result = extract_pnad_zip(zip_path, extracted_base_dir=tmp_path / "extracted")

    assert result.extracted is False
    assert result.extracted_files == [existing_file]
    assert existing_file.read_text() == "existente"


def test_extract_pnad_zip_force_reextracts_existing_output_dir(tmp_path):
    zip_path = tmp_path / "PNADC_2023_trimestre1.zip"
    create_test_zip(zip_path)
    output_dir = tmp_path / "extracted" / "PNADC_2023_trimestre1"
    output_dir.mkdir(parents=True)
    existing_file = output_dir / "dados.txt"
    existing_file.write_text("existente")

    result = extract_pnad_zip(
        zip_path,
        extracted_base_dir=tmp_path / "extracted",
        force=True,
    )

    assert result.extracted is True
    assert existing_file.read_text() == "conteudo"


def test_extract_pnad_zip_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pnad_zip(tmp_path / "missing.zip", extracted_base_dir=tmp_path)
