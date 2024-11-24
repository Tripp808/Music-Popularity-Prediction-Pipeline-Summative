# python runtime
FROM python:3.10-slim

# working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt from the 'src' directory into the container
COPY src/requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# where to find my modules in 'src'
ENV PYTHONPATH=/app/src

# Copied 'src' folder to the container's /app/src directory
COPY src/ /app/src/

# Expose port 8000 for the app
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
