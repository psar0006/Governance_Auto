import pandas as pd
from sqlalchemy import create_engine
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFrameProcessor:
    """
    A utility class for processing, profiling, and exporting DataFrames with 
    data dictionaries, QA checks, and basic lineage comparison.
    """

    def __init__(self, df=None):
        """Initialize the processor with an optional DataFrame."""
        self.df = df

    # -------------------------------
    # Data Ingestion Methods
    # -------------------------------
    @staticmethod
    def load_dataset(file_path):
        """Load a CSV file into a DataFrame."""
        if not file_path.endswith(".csv"):
            raise ValueError("Only CSV files are supported.")
        try:
            df = pd.read_csv(file_path)
            return DataFrameProcessor(df)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    @staticmethod
    def load_excel(file_path, sheet_name=0):
        """Load an Excel file into a DataFrame."""
        if not file_path.endswith(('.xls', '.xlsx')):
            raise ValueError("Only Excel files are supported.")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return DataFrameProcessor(df)
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise

    @staticmethod
    def load_sql_table(conn_str, table_name):
        """Load a SQL table into a DataFrame."""
        try:
            engine = create_engine(conn_str)
            df = pd.read_sql_table(table_name, engine)
            return DataFrameProcessor(df)
        except Exception as e:
            logger.error(f"Failed to load SQL table: {e}")
            raise

    def set_dataframe(self, df):
        """Attach a DataFrame to the processor instance."""
        self.df = df
        return self  # enable chaining

    # -------------------------------
    # Metadata: Data Dictionary
    # -------------------------------
    def generate_data_dictionary(self):
        """Generate a basic data dictionary for the DataFrame."""
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        dictionary = []
        for col in self.df.columns:
            series = self.df[col]
            col_info = {
                "column_name": col,
                "dtype": str(series.dtype),
                "num_nulls": int(series.isnull().sum()),
                "num_distinct": int(series.nunique()),
                "example_values": series.dropna().unique()[:5].tolist()
            }
            dictionary.append(col_info)
        return dictionary

    # -------------------------------
    # Data Quality / QA Checks
    # -------------------------------
    def qa_checks(self, include_top_values=True):
        """
        Run standard QA checks on DataFrame columns.

        Returns:
            dict: QA statistics per column.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        checks = {}
        for col in self.df.columns:
            series = self.df[col]
            col_checks = {
                "null_pct": round(series.isnull().mean() * 100, 2),
                "col_duplicates": int(series.duplicated().sum()),
            }

            # Numeric stats
            if pd.api.types.is_numeric_dtype(series):
                col_checks.update({
                    "min": float(series.min()) if not series.empty else None,
                    "max": float(series.max()) if not series.empty else None,
                    "mean": float(series.mean()) if not series.empty else None
                })

            # Top values for categorical/object columns
            if include_top_values:
                col_checks["top_values"] = series.value_counts().head(3).to_dict()

            checks[col] = col_checks

        # Row-level duplicates (only once)
        checks["_row_duplicates"] = int(self.df.duplicated().sum())
        return checks

    # -------------------------------
    # Lineage Comparison
    # -------------------------------
    def compare_lineage(self, other_df):
        """
        Compare columns with another DataFrame to get basic lineage.

        Args:
            other_df (pd.DataFrame): Another DataFrame to compare.

        Returns:
            dict: Common, new, and removed columns.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        return {
            "common_columns": list(set(self.df.columns) & set(other_df.columns)),
            "new_columns": list(set(other_df.columns) - set(self.df.columns)),
            "removed_columns": list(set(self.df.columns) - set(other_df.columns))
        }

    # -------------------------------
    # Export Utilities
    # -------------------------------
    def export_to_json(self, output_path="report.json"):
        """Export data dictionary + QA checks to a JSON file."""
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        report = {
            "data_dictionary": self.generate_data_dictionary(),
            "qa_checks": self.qa_checks()
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
        return output_path

    def export_to_csv(self, output_path="data_dictionary.csv", include_qa=True):
        """
        Export data dictionary (optionally including QA checks) to CSV.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        data_dict = self.generate_data_dictionary()

        if include_qa:
            qa = self.qa_checks(include_top_values=False)
            for col_info in data_dict:
                col_name = col_info["column_name"]
                if col_name in qa:
                    col_info.update(qa[col_name])

        df_out = pd.DataFrame(data_dict)
        df_out.to_csv(output_path, index=False)
        return output_path

    def export_to_xlsx(self, output_path="data_dictionary.xlsx", include_qa=True):
        """
        Export data dictionary and QA checks to Excel, optionally including QA.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded.")

        dict_df = pd.DataFrame(self.generate_data_dictionary())

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            dict_df.to_excel(writer, sheet_name="Data Dictionary", index=False)

            if include_qa:
                qa = self.qa_checks()
                qa_df = pd.DataFrame.from_dict(qa, orient="index").reset_index()
                qa_df.rename(columns={"index": "column_name"}, inplace=True)
                qa_df.to_excel(writer, sheet_name="QA Checks", index=False)

        return output_path

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "salary": [50000, 60000, 75000, 80000],
        "department": ["HR", "Finance", "Finance", "IT"]
    }

    processor = DataFrameProcessor(pd.DataFrame(data))

    # Generate dictionary
    print("Data Dictionary:")
    print(processor.generate_data_dictionary())

    # Run QA
    print("\nQA Checks:")
    print(processor.qa_checks())

    # Compare lineage with another DF
    other_df = pd.DataFrame({
        "name": ["Eve"],
        "age": [28],
        "location": ["Sydney"]
    })
    print("\nLineage Comparison:")
    print(processor.compare_lineage(other_df))

    # Export reports
    processor.export_to_json("report.json")
    processor.export_to_csv("report.csv")
    processor.export_to_xlsx("report.xlsx")
    print("\nExports complete.")
