import importlib


def test_inspect_pnad_file_imports_without_reading_data():
    module = importlib.import_module("scripts.inspect_pnad_file")

    assert module.REPORT_PATH.name == "pnad_inspection_report.txt"
    assert callable(module.read_pnad_file)
    assert callable(module.build_report)
    assert callable(module.inspect_file)
