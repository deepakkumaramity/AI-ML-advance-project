# Lightweight PyTorch + FastAPI runtime
FROM python:3.10-slim

WORKDIR /app

# System deps for torchvision / opencv
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    libgl1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
