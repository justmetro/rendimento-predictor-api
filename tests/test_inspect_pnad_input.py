from scripts.inspect_pnad_input import (
    find_relevant_lines,
    read_text_with_fallback,
)


def test_read_text_with_fallback_reads_latin1_file(tmp_path):
    file_path = tmp_path / "input.txt"
    file_path.write_bytes("posição inicial".encode("latin1"))

    text, encoding = read_text_with_fallback(file_path)

    assert text == "posição inicial"
    assert encoding == "latin1"


def test_find_relevant_lines_matches_terms_case_insensitive():
    lines = [
        "Variavel sem interesse",
        "V2009 idade do morador",
        "Descricao da uf",
        "VD4019 rendimento efetivo",
    ]

    result = find_relevant_lines(lines, ("V2009", "UF", "VD4019"))

    assert result == [
        "2: V2009 idade do morador",
        "3: Descricao da uf",
        "4: VD4019 rendimento efetivo",
    ]
