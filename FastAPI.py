from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import tempfile
from dataframe_processor import DataFrameProcessor

app = FastAPI(
    title="Audit-Ready Data Governance API",
    description="Lightweight SaaS backend that auto-generates data dictionaries, lineage reports, and QA checks.",
    version="1.0.0"
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset and return a basic profile."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        return JSONResponse({"error": "Only CSV or Excel files supported."}, status_code=400)
    
    # Save file temporarily
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()
    
    # Load dataframe
    if file.filename.endswith(".csv"):
        df = pd.read_csv(temp.name)
    else:
        df = pd.read_excel(temp.name)
    
    processor = DataFrameProcessor(df)
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "preview": df.head(5).to_dict(orient="records")
    }
    return {"message": "File uploaded successfully.", "summary": summary}

@app.post("/data-dictionary")
async def get_data_dictionary(file: UploadFile = File(...)):
    """Generate data dictionary for uploaded dataset."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()
    df = pd.read_csv(temp.name)
    processor = DataFrameProcessor(df)
    return processor.generate_data_dictionary()

@app.post("/qa-checks")
async def get_qa_checks(file: UploadFile = File(...)):
    """Run QA checks for uploaded dataset."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()
    df = pd.read_csv(temp.name)
    processor = DataFrameProcessor(df)
    return processor.qa_checks()

@app.post("/export/xlsx")
async def export_to_excel(file: UploadFile = File(...)):
    """Generate full audit report in Excel."""
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()
    df = pd.read_csv(temp.name)
    processor = DataFrameProcessor(df)
    output_path = processor.export_to_xlsx("audit_report.xlsx")
    return FileResponse(output_path, filename="audit_report.xlsx")

