FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .
CMD ["uvicorn","ai_gateway.mcp:app","--host","0.0.0.0","--port","8200"]
