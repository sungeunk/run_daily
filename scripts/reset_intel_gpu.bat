@echo off
setlocal EnableExtensions DisableDelayedExpansion

:: Reset Intel display device(s) without reboot.
:: Usage:
::   reset_intel_gpu.bat [wait_seconds]
:: Examples:
::   reset_intel_gpu.bat
::   reset_intel_gpu.bat 30
:: Internal options (edit below in this script):
::   GPU_FILTER=Intel
::   MEM_CLEANUP=1  (1=enabled, 0=disabled)

set "WAIT_SECS=%~1"
if "%WAIT_SECS%"=="" set "WAIT_SECS=20"

:: Internal configuration
set "GPU_FILTER=Intel"
set "MEM_CLEANUP=1"

echo [INFO] Requested wait time: %WAIT_SECS%s
echo [INFO] Device name filter: %GPU_FILTER%
echo [INFO] Memory cleanup (0/1): %MEM_CLEANUP%

:: Require admin privileges.
net session >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    echo [ERROR] Administrator privilege is required.
    echo [HINT] Right-click CMD and select "Run as administrator".
    exit /b 1
)

where pnputil >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    echo [ERROR] pnputil was not found.
    exit /b 1
)

echo [INFO] Finding display devices that match "%GPU_FILTER%"...
set "TMP_IDS=%TEMP%\intel_gpu_ids_%RANDOM%_%RANDOM%.txt"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$f = [regex]::Escape('%GPU_FILTER%');" ^
  "Get-PnpDevice -Class Display ^| Where-Object { $_.FriendlyName -match $f } ^|" ^
  "Select-Object -ExpandProperty InstanceId" > "%TMP_IDS%"

findstr /r /c:"." "%TMP_IDS%" >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    echo [ERROR] No display device matched "%GPU_FILTER%".
    echo [INFO] Available display devices:
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-PnpDevice -Class Display ^| Select-Object Status,FriendlyName,InstanceId ^| Format-Table -AutoSize"
    del /q "%TMP_IDS%" >nul 2>&1
    exit /b 2
)

set /a TOTAL=0
set /a OK=0

for /f "usebackq delims=" %%I in ("%TMP_IDS%") do (
    set /a TOTAL+=1
    echo.
    echo [INFO] Restarting device: %%I

    pnputil /restart-device "%%I" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] restart-device failed. Trying disable/enable fallback...
        pnputil /disable-device "%%I" >nul 2>&1
        if errorlevel 1 (
            echo [FAIL] disable-device failed for: %%I
        ) else (
            timeout /t 2 /nobreak >nul
            pnputil /enable-device "%%I" >nul 2>&1
            if errorlevel 1 (
                echo [FAIL] enable-device failed for: %%I
            ) else (
                echo [ OK ] disable/enable fallback succeeded.
                set /a OK+=1
            )
        )
    ) else (
        echo [ OK ] restart-device succeeded.
        set /a OK+=1
    )
)

del /q "%TMP_IDS%" >nul 2>&1

if "%MEM_CLEANUP%"=="1" (
    setlocal EnableDelayedExpansion
    echo.
    echo [INFO] Running memory-state cleanup...

    set "EMPTY_STANDBY_TOOL="

    where /q EmptyStandbyList.exe
    if "%ERRORLEVEL%"=="0" (
        set "EMPTY_STANDBY_TOOL=EmptyStandbyList.exe"
    )

    if "!EMPTY_STANDBY_TOOL!"=="" (
        if exist "%~dp0EmptyStandbyList.exe" (
            set "EMPTY_STANDBY_TOOL=%~dp0EmptyStandbyList.exe"
        )
    )

    if "!EMPTY_STANDBY_TOOL!"=="" (
        if exist "%~dp0tools\EmptyStandbyList.exe" (
            set "EMPTY_STANDBY_TOOL=%~dp0tools\EmptyStandbyList.exe"
        )
    )

    if "!EMPTY_STANDBY_TOOL!"=="" (
        if exist "C:\Tools\EmptyStandbyList.exe" (
            set "EMPTY_STANDBY_TOOL=C:\Tools\EmptyStandbyList.exe"
        )
    )

    if "!EMPTY_STANDBY_TOOL!"=="" (
        echo [WARN] EmptyStandbyList.exe not found. Skipping standby cleanup.
        echo [HINT] Put EmptyStandbyList.exe in PATH, "%~dp0", or "%~dp0tools".
    ) else (
        echo [INFO] Using: !EMPTY_STANDBY_TOOL!
        "!EMPTY_STANDBY_TOOL!" workingsets >nul 2>&1
        "!EMPTY_STANDBY_TOOL!" modifiedpagelist >nul 2>&1
        "!EMPTY_STANDBY_TOOL!" standbylist >nul 2>&1
        echo [INFO] Memory-state cleanup finished.
    )
    endlocal
)

echo.
echo [INFO] Reset summary: %OK% / %TOTAL% device(s) succeeded.
echo [INFO] Waiting %WAIT_SECS%s for driver stabilization...
timeout /t %WAIT_SECS% /nobreak >nul

echo [INFO] Done. You can now run benchmark.py.
exit /b 0
