FROM python:3.10-slim

# my working directory 
WORKDIR /app

# Copy of requirements.txt from the 'src' directory into the container
COPY src/requirements.txt /app/requirements.txt

# dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy of everythig from the 'src' directory into /app/src within the container
COPY src/ /app/src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
