@echo off
echo Starting TSL-Translator FastAPI Server...
echo.
echo Server will be available at:
echo   - Local: http://localhost:8000
echo   - Network: http://192.168.1.11:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload