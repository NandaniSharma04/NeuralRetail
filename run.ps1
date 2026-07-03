# NeuralRetail Backend Runner

Write-Host "Starting NeuralRetail Backend (FastAPI)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\.venv\Scripts\activate; uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

Write-Host "Done! Backend is running on http://127.0.0.1:8000" -ForegroundColor Cyan
