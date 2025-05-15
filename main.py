from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os

from screenplay_pdf_to_json.parse_pdf import parsePdf, groupDualDialogues, groupSections, sortLines, cleanPage, getTopTrends, stitchSeperateWordsIntoLines, processInitialPages
from screenplay_pdf_to_json.utils import cleanScript

app = FastAPI(title="Screenplay PDF to JSON Converter API",
              description="API to convert screenplay PDFs into structured JSON format")

@app.post("/convert/", response_class=JSONResponse)
async def convert_pdf_to_json(
    pdf_file: UploadFile = File(...),
    page_start: int = None
):
    """
    Convert a screenplay PDF file to JSON format.
    
    - **pdf_file**: The screenplay PDF file to convert
    - **page_start**: Optional page number to start parsing from
    """
    # Validate file type
    if not pdf_file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        # Write the uploaded file content to the temporary file
        temp_pdf.write(await pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    try:
        # Open the temporary PDF file for processing
        with open(temp_pdf_path, 'rb') as script_file:
            # Process the PDF file
            result = convert(script_file, page_start)
        
        # Delete the temporary file after processing
        os.unlink(temp_pdf_path)
        
        # Return the JSON result
        return result
    
    except Exception as e:
        # Delete the temporary file in case of error
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        
        # Raise an HTTP exception with the error details
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF: {str(e)}")

def convert(script_file, page_start):
    """
    Convert a screenplay PDF file to JSON format.
    
    Args:
        script_file: The PDF file object
        page_start: Optional page number to start parsing from
        
    Returns:
        A list of screenplay elements in JSON format
    """
    # Parse script based on pdfminer.six
    new_script = parsePdf(script_file)["pdf"]

    # Process initial pages
    first_pages_dict = processInitialPages(new_script)
    first_pages = first_pages_dict["firstPages"]
    skip_page = page_start if page_start else first_pages_dict["pageStart"]

    # Remove any useless line (page number, empty line, special symbols)
    new_script = cleanPage(new_script, skip_page)

    # Sort lines by y. If y is the same, then sort by x
    new_script = sortLines(new_script, skip_page)

    # Group dual dialogues into the same segments
    new_script = groupDualDialogues(new_script, skip_page)

    # Stitch words into what's supposed to be part of the same line
    new_script = stitchSeperateWordsIntoLines(new_script, skip_page)
    
    # Get top trends
    top_trends = getTopTrends(new_script)

    # Group into sections based on type
    new_script = groupSections(top_trends, new_script, skip_page, False)

    # Clean the script
    new_script = cleanScript(new_script, False)

    # Add first pages
    new_script = first_pages + new_script
    
    return new_script

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)