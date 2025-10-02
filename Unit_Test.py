import os
import pandas as pd
import pytest
from dataframe_processor import DataFrameProcessor

# -------------------------------
# Sample DataFrame for testing
# -------------------------------
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David", None],
    "age": [25, 30, 35, 40, 28],
    "salary": [50000, 60000, 75000, 80000, 72000],
    "department": ["HR", "Finance", "Finance", "IT", "HR"]
})

# -------------------------------
# Ingestion Tests
# -------------------------------
def test_set_dataframe():
    processor = DataFrameProcessor()
    processor.set_dataframe(df)
    pd.testing.assert_frame_equal(processor.df, df)

def test_load_dataset(tmp_path):
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    processor = DataFrameProcessor.load_dataset(file_path)
    pd.testing.assert_frame_equal(processor.df, df)

def test_load_excel(tmp_path):
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    processor = DataFrameProcessor.load_excel(file_path)
    pd.testing.assert_frame_equal(processor.df, df)

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
    # Column-level checks
    for col in df.columns:
        assert "null_pct" in qa[col]
        assert "col_duplicates" in qa[col]
        assert "top_values" in qa[col]
        if pd.api.types.is_numeric_dtype(df[col]):
            assert "min" in qa[col] and "max" in qa[col] and "mean" in qa[col]
    # Row-level duplicates
    assert "_row_duplicates" in qa

# -------------------------------
# Lineage Tests
# -------------------------------
def test_compare_lineage():
    processor = DataFrameProcessor(df)
    other_df = pd.DataFrame({
        "name": ["Eve"],
        "age": [28],
        "location": ["Sydney"]
    })
    lineage = processor.compare_lineage(other_df)
    assert set(lineage.keys()) == {"common_columns", "new_columns", "removed_columns"}
    assert "name" in lineage["common_columns"]
    assert "location" in lineage["new_columns"]
    assert "department" in lineage["removed_columns"]

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
    assert "data_dictionary" in report
    assert "qa_checks" in report

def test_export_to_csv(tmp_path):
    processor = DataFrameProcessor(df)
    output_file = tmp_path / "report.csv"
    path = processor.export_to_csv(output_file)
    assert os.path.exists(path)
    exported_df = pd.read_csv(path)
    assert set(["column_name", "null_pct", "col_duplicates"]) <= set(exported_df.columns)

def test_export_to_xlsx(tmp_path):
    processor = DataFrameProcessor(df)
    output_file = tmp_path / "report.xlsx"
    path = processor.export_to_xlsx(output_file)
    assert os.path.exists(path)
    xls = pd.ExcelFile(path)
    assert "Data Dictionary" in xls.sheet_names
    assert "QA Checks" in xls.sheet_names
