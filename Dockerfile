# Stage 1: Build
FROM python:3.10-slim as builder
RUN apt-get update && apt-get install -y gcc g++ build-essential libgomp1
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (smaller!)
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src ./src
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]