import pandas as pd

from scripts.parse_pnad_real import FINAL_COLUMNS, parse_pnad_file


def build_fixed_width_line(values):
    line = [" "] * 560

    fields = {
        "UF": (5, 7),
        "V2007": (94, 95),
        "V2009": (103, 106),
        "V2010": (106, 107),
        "VD3005": (507, 509),
        "VD4010": (519, 521),
        "VD4019": (544, 552),
        "VD4020": (552, 560),
    }

    for name, (start, end) in fields.items():
        value = str(values[name]).rjust(end - start)
        line[start:end] = value

    return "".join(line)


def test_parse_pnad_file_generates_standardized_columns_and_mappings(tmp_path):
    raw_path = tmp_path / "PNADC_2023_trimestre1.txt"
    output_path = tmp_path / "processed.csv"
    raw_path.write_text(
        "\n".join(
            [
                build_fixed_width_line(
                    {
                        "UF": "33",
                        "V2007": "1",
                        "V2009": "35",
                        "V2010": "4",
                        "VD3005": "12",
                        "VD4010": "06",
                        "VD4019": "00003200",
                        "VD4020": "00000000",
                    }
                ),
                build_fixed_width_line(
                    {
                        "UF": "41",
                        "V2007": "2",
                        "V2009": "42",
                        "V2010": "1",
                        "VD3005": "16",
                        "VD4010": "03",
                        "VD4019": "00000000",
                        "VD4020": "00004800",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    df, rows_read = parse_pnad_file(raw_path, output_path=output_path)

    assert rows_read == 2
    assert list(df.columns) == FINAL_COLUMNS
    assert df.to_dict(orient="records") == [
        {
            "idade": 35,
            "sexo": "M",
            "cor_raca": "Parda",
            "anos_estudo": 12,
            "setor": "Comercio",
            "regiao": "Sudeste",
            "rendimento_hora": 20.0,
        },
        {
            "idade": 42,
            "sexo": "F",
            "cor_raca": "Branca",
            "anos_estudo": 16,
            "setor": "Industria",
            "regiao": "Sul",
            "rendimento_hora": 30.0,
        },
    ]

    saved_df = pd.read_csv(output_path)
    assert list(saved_df.columns) == FINAL_COLUMNS


def test_parse_pnad_file_respects_sample_size(tmp_path):
    raw_path = tmp_path / "PNADC_2023_trimestre1.txt"
    output_path = tmp_path / "processed.csv"
    raw_path.write_text(
        "\n".join(
            [
                build_fixed_width_line(
                    {
                        "UF": "33",
                        "V2007": "1",
                        "V2009": "35",
                        "V2010": "4",
                        "VD3005": "12",
                        "VD4010": "06",
                        "VD4019": "00003200",
                        "VD4020": "00000000",
                    }
                ),
                build_fixed_width_line(
                    {
                        "UF": "41",
                        "V2007": "2",
                        "V2009": "42",
                        "V2010": "1",
                        "VD3005": "16",
                        "VD4010": "03",
                        "VD4019": "00004800",
                        "VD4020": "00000000",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    df, rows_read = parse_pnad_file(
        raw_path,
        output_path=output_path,
        sample_size=1,
    )

    assert rows_read == 1
    assert df.shape[0] == 1
