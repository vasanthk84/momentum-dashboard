@echo off
title Momentum Scanner
cd /d "%~dp0"

echo.
echo  +------------------------------------------+
echo  ^|      Momentum Scanner  -  NSE             ^|
echo  +------------------------------------------+
echo.

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Node.js not found. Please install from https://nodejs.org
    pause & exit /b 1
)

if not exist "node_modules\" (
    echo  [SETUP] Installing dependencies...
    call npm install
    if %errorlevel% neq 0 ( echo  [ERROR] npm install failed. & pause & exit /b 1 )
    echo  [SETUP] Dependencies installed.
    echo.
)

echo  Clearing ports 3000 and 5173...
powershell -NoProfile -Command "$p=(Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue).OwningProcess; if($p){Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Write-Host '  Killed PID' $p 'on port 3000'}"
powershell -NoProfile -Command "$p=(Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue).OwningProcess; if($p){Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Write-Host '  Killed PID' $p 'on port 5173'}"
echo.

echo  Choose launch mode:
echo.
echo    [1]  Production  (npm run build + serve at :3000)
echo    [2]  Dev         (hot-reload Vite :5173 + API :3000)
echo.
set /p MODE=" Enter 1 or 2: "

if "%MODE%"=="1" goto PRODUCTION
if "%MODE%"=="2" goto DEV
echo  Invalid choice. Defaulting to Dev mode.
goto DEV

:PRODUCTION
echo.
echo  [BUILD] Building React app...
call npm run build
if %errorlevel% neq 0 ( echo  [ERROR] Build failed. & pause & exit /b 1 )
echo  [BUILD] Done.
echo.
echo  [START] Starting server at http://localhost:3000
start "" "http://localhost:3000"
node server.js
pause
goto END

:DEV
echo.
echo  Starting dev servers in separate windows...
echo.
start "MS - API Server (:3000)" cmd /k "node server.js"
timeout /t 2 /nobreak >nul
start "MS - Vite Dev  (:5173)" cmd /k "npx vite"
timeout /t 3 /nobreak >nul
echo  [READY] Opening http://localhost:5173
start "" "http://localhost:5173"
echo.
echo  API:  http://localhost:3000
echo  UI:   http://localhost:5173
echo.
echo  Close this window or the server windows to stop.
pause

:END
