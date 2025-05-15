FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "convert.py"]
