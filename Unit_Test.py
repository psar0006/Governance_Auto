import os
import pandas as pd
import pytest
from dataframe_processor import DataFrameProcessor

# -------------------------------
# Sample DataFrame for testing
# -------------------------------
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", None],
    "age": [25, 30, 35, 40],
    "salary": [50000, 60000, 75000, 80000],
    "department": ["HR", "Finance", "Finance", "IT"]
})

# -------------------------------
# Ingestion Tests
# -------------------------------
def test_set_dataframe():
    processor = DataFrameProcessor()
    processor.set_dataframe(df)
    assert processor.df.equals(df)

def test_load_dataset(tmp_path):
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    loaded_df = DataFrameProcessor.load_dataset(file_path)
    assert loaded_df.equals(df)

def test_load_excel(tmp_path):
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    loaded_df = DataFrameProcessor.load_excel(file_path)
    pd.testing.assert_frame_equal(loaded_df, df)

# -------------------------------
# Data Dictionary & QA Tests
# -------------------------------
def test_generate_data_dictionary():
    processor = DataFrameProcessor(df)
    dictionary = processor.generate_data_dictionary()
    assert isinstance(dictionary, list)
    assert all("column_name" in col for col in dictionary)
    assert len(dictionary) == df.shape[1]

def test_qa_checks():
    processor = DataFrameProcessor(df)
    qa = processor.qa_checks()
    assert isinstance(qa, dict)
    for col in df.columns:
        assert "null_pct" in qa[col]
        assert "duplicate_count" in qa[col]
        if pd.api.types.is_numeric_dtype(df[col]):
            assert "min" in qa[col] and "max" in qa[col] and "mean" in qa[col]

# -------------------------------
# Export Tests
# -------------------------------
def test_export_to_json(tmp_path):
    processor = DataFrameProcessor(df)
    output_file = tmp_path / "report.json"
    path = processor.export_to_json(output_file)
    assert os.path.exists(path)
    with open(path) as f:
        report = pd.json.load(f)
    assert "data_dictionary" in report and "qa_checks" in report

def test_export_to_csv(tmp_path):
    processor = DataFrameProcessor(df)
    output_file = tmp_path / "report.csv"
    path = processor.export_to_csv(output_file)
    assert os.path.exists(path)
    exported_df = pd.read_csv(path)
    assert set(["column_name", "null_pct"]) <= set(exported_df.columns)

def test_export_to_xlsx(tmp_path):
    processor = DataFrameProcessor(df)
    output_file = tmp_path / "report.xlsx"
    path = processor.export_to_xlsx(output_file)
    assert os.path.exists(path)
    # Ensure sheets exist
    xls = pd.ExcelFile(path)
    assert "Data Dictionary" in xls.sheet_names
    assert "QA Checks" in xls.sheet_names
