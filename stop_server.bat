@echo off
:: إيقاف RFP Intelligence Server

echo Stopping RFP Intelligence Server on port 8000...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr LISTENING') do (
    echo Found process PID: %%a
    taskkill /PID %%a /F
    echo Server stopped.
    goto :done
)
echo No server found running on port 8000.
:done
