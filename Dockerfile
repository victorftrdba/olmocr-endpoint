FROM alleninstituteforai/olmocr:latest

WORKDIR /app
COPY main.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]