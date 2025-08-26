import pandas as pd
from sqlalchemy import create_engine
import json

class DataFrameProcessor:
    def __init__(self, df=None):
        """Initialize with an optional dataframe."""
        self.df = df

    # -------------------------------
    # Ingestion Methods
    # -------------------------------
    @staticmethod
    def load_dataset(file_path):
        """Load a CSV file into a DataFrame."""
        if not file_path.endswith('.csv'):
            raise ValueError("Only CSV files are supported.")
        return pd.read_csv(file_path)

    @staticmethod
    def load_sql_table(conn_str, table_name):
        """Load a SQL table into a DataFrame."""
        engine = create_engine(conn_str)
        return pd.read_sql_table(table_name, engine)
    
    @staticmethod
    def load_excel(file_path, sheet_name=0):
        """Load an Excel file into a DataFrame."""
        if not file_path.endswith(('.xls', '.xlsx')):
            raise ValueError("Only Excel files are supported.")
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def set_dataframe(self, df):
        """Attach a DataFrame to the processor instance."""
        self.df = df

    # -------------------------------
    # Metadata: Data Dictionary
    # -------------------------------
    def generate_data_dictionary(self):
        """Generate a basic data dictionary for the dataframe."""
        if self.df is None:
            raise ValueError("No dataframe loaded.")
        
        dictionary = []
        for col in self.df.columns:
            col_info = {
                "column_name": col,
                "dtype": str(self.df[col].dtype),
                "num_nulls": int(self.df[col].isnull().sum()),
                "num_distinct": int(self.df[col].nunique()),
                "example_values": self.df[col].dropna().unique()[:5].tolist()
            }
            dictionary.append(col_info)
        return dictionary

    # -------------------------------
    # Data Quality / QA Checks
    # -------------------------------
    def qa_checks(self):
        """Run standard QA checks on dataframe columns."""
        if self.df is None:
            raise ValueError("No dataframe loaded.")

        checks = {}
        for col in self.df.columns:
            series = self.df[col]
            checks[col] = {
                "null_pct": round(series.isnull().mean() * 100, 2),
                "duplicate_count": int(self.df.duplicated(subset=[col]).sum())
            }
            # Only run numeric checks on number columns
            if pd.api.types.is_numeric_dtype(series):
                checks[col].update({
                    "min": float(series.min()) if not series.empty else None,
                    "max": float(series.max()) if not series.empty else None,
                    "mean": float(series.mean()) if not series.empty else None
                })
        return checks

    # -------------------------------
    # Export Utilities
    # -------------------------------
    def export_to_json(self, output_path="report.json"):
        """Export dictionary + QA results to JSON file."""
        if self.df is None:
            raise ValueError("No dataframe loaded.")

        report = {
            "data_dictionary": self.generate_data_dictionary(),
            "qa_checks": self.qa_checks()
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
        return output_path

    def export_to_csv(self, output_path="data_dictionary.csv"):
        "Export data dictionary + QA checks to CSV file."
        if self.df is None:
            raise ValueError("No dataframe loaded.")

        data_dict = self.generate_data_dictionary()
        qa = self.qa_checks()

        # Merge QA results into data dictionary
        for col_info in data_dict:
            col_name = col_info["column_name"]
            if col_name in qa:
                col_info.update(qa[col_name])  # add QA stats into dict

        dict_df = pd.DataFrame(data_dict)
        dict_df.to_csv(output_path, index=False)
        return output_path


    def export_to_xlsx(self, output_path="data_dictionary.xlsx"):
        """Export data dictionary and QA checks to separate sheets in Excel."""
        if self.df is None:
            raise ValueError("No dataframe loaded.")

        # Generate outputs
        data_dict = self.generate_data_dictionary()
        qa = self.qa_checks()

        # Convert QA dict (col → stats) into DataFrame
        qa_df = pd.DataFrame.from_dict(qa, orient="index").reset_index()
        qa_df.rename(columns={"index": "column_name"}, inplace=True)

        # Convert data dictionary list → DataFrame
        dict_df = pd.DataFrame(data_dict)

        # Write both to Excel
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            dict_df.to_excel(writer, sheet_name="Data Dictionary", index=False)
            qa_df.to_excel(writer, sheet_name="QA Checks", index=False)

        return output_path
