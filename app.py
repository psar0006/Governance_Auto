from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import os
from DataFrame_Processer import DataFrameProcessor

app = FastAPI(
    title="Audit-Ready Data Governance API",
    description="Lightweight SaaS backend that auto-generates data dictionaries, lineage reports, and QA checks.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file(file: UploadFile) -> bool:
    """Validate file type."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in ALLOWED_EXTENSIONS

async def load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Load dataframe from uploaded file with error handling."""
    try:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size exceeds 50MB limit.")
        
        temp.write(contents)
        temp.close()
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(temp.name)
        else:  # .xlsx or .xls
            df = pd.read_excel(temp.name)
        
        # Cleanup temp file
        os.unlink(temp.name)
        return df
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset and return a basic profile."""
    if not validate_file(file):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    df = await load_dataframe(file)
    
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }
    return {"message": "File uploaded successfully.", "summary": summary}

@app.post("/data-dictionary")
async def get_data_dictionary(file: UploadFile = File(...)):
    """Generate data dictionary for uploaded dataset."""
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    df = await load_dataframe(file)
    processor = DataFrameProcessor(df)
    
    try:
        dictionary = processor.generate_data_dictionary()
        return {
            "data_dictionary": dictionary,
            "total_columns": len(dictionary)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating data dictionary: {str(e)}")

@app.post("/qa-checks")
async def get_qa_checks(file: UploadFile = File(...)):
    """Run QA checks for uploaded dataset."""
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    df = await load_dataframe(file)
    processor = DataFrameProcessor(df)
    
    try:
        checks = processor.qa_checks()
        return {
            "qa_checks": checks,
            "row_duplicates": checks.pop("_row_duplicates", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running QA checks: {str(e)}")

@app.post("/export/xlsx")
async def export_to_excel(file: UploadFile = File(...)):
    """Generate full audit report in Excel."""
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    df = await load_dataframe(file)
    processor = DataFrameProcessor(df)
    
    try:
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        temp_output.close()
        
        output_path = processor.export_to_xlsx(temp_output.name, include_qa=True)
        
        return FileResponse(
            output_path,
            filename="audit_report.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting Excel: {str(e)}")

@app.post("/export/json")
async def export_to_json(file: UploadFile = File(...)):
    """Generate full audit report in JSON."""
    if not validate_file(file):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    df = await load_dataframe(file)
    processor = DataFrameProcessor(df)
    
    try:
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_output.close()
        
        output_path = processor.export_to_json(temp_output.name)
        
        return FileResponse(
            output_path,
            filename="audit_report.json",
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting JSON: {str(e)}")

@app.post("/lineage-compare")
async def compare_lineage(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare two datasets for column lineage."""
    if not (validate_file(file1) and validate_file(file2)):
        raise HTTPException(status_code=400, detail="Invalid file formats.")
    
    try:
        df1 = await load_dataframe(file1)
        df2 = await load_dataframe(file2)
        
        processor = DataFrameProcessor(df1)
        comparison = processor.compare_lineage(df2)
        
        return {
            "file1": file1.filename,
            "file2": file2.filename,
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing lineage: {str(e)}")
