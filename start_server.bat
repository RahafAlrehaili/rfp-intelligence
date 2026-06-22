@echo off
:: RFP Intelligence Server — يشغّل الـ server في نافذة مستقلة
:: يستمر حتى لو أُغلق الـ terminal الأصلي

set PROJ_DIR=%~dp0
set PYTHON=python
set PORT=8000
set PYTHONIOENCODING=utf-8

echo.
echo  Starting RFP Intelligence Server on http://127.0.0.1:%PORT%
echo  Project: %PROJ_DIR%
echo.

:: شغّل في نافذة جديدة — تبقى مفتوحة حتى لو أُغلق هذا الـ terminal
start "RFP Intelligence Server" /D "%PROJ_DIR%" %PYTHON% -m uvicorn src.api:app --host 127.0.0.1 --port %PORT% --no-access-log

timeout /t 3 /nobreak >nul
echo  Server started. Check the new window for logs.
echo  To stop: close the server window or run stop_server.bat
echo.
