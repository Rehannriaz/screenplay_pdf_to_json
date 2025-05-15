# Screenplay PDF to JSON API

This FastAPI application converts screenplay PDFs into structured JSON format.

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the API

Run the API server with:

```bash
python main.py
```

This will start the FastAPI server on http://0.0.0.0:8000.

## API Usage

### Convert PDF to JSON

**Endpoint**: `POST /convert/`

**Parameters**:
- `pdf_file`: The screenplay PDF file to convert (required)
- `page_start`: Optional page number to start parsing from (optional)

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/convert/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "pdf_file=@path/to/screenplay.pdf" \
  -F "page_start=1"
```

**Example using Python requests**:
```python
import requests

url = "http://localhost:8000/convert/"
files = {"pdf_file": open("path/to/screenplay.pdf", "rb")}
data = {"page_start": 1}  # Optional

response = requests.post(url, files=files, data=data)
json_data = response.json()
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

- `main.py`: FastAPI application entry point
- `requirements.txt`: List of dependencies
- `screenplay_pdf_to_json/`: Your screenplay parsing package

## Error Handling

The API will return appropriate HTTP status codes:
- `400`: Bad Request (e.g., non-PDF file submitted)
- `500`: Internal Server Error (parsing failure)
