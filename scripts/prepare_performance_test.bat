@echo off
setlocal

echo ========================================
echo  LLM Benchmark Performance Preparation
echo ========================================

:: ----------------------------------------
:: Require Administrator privileges
:: ----------------------------------------

net session >nul 2>&1 || (
    echo [ERROR] This script must be run as Administrator.
    exit /b 1
)

:: ----------------------------------------
:: Power policy: Ultimate Performance
:: ----------------------------------------

echo [1/6] Setting power policy...

powercfg /S SCHEME_MIN

set ULTIMATE_APPLIED=0
for /f "tokens=2" %%i in ('powercfg -list ^| findstr /i "Ultimate"') do (
    powercfg /S %%i
    set ULTIMATE_APPLIED=1
    echo       Ultimate Performance applied: %%i
)
if %ULTIMATE_APPLIED%==0 echo [WARN] Ultimate Performance plan not found, using High Performance.

powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100
powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100
powercfg -setacvalueindex scheme_current sub_processor CPMINCORES 100
powercfg -setactive scheme_current

:: ----------------------------------------
:: Game Mode + HAGS
:: ----------------------------------------

echo [2/6] Enabling Game Mode and HAGS...

:: NOTE: Game Mode OFF — compute workloads (LLM inference) are not recognized as games;
::       enabling it can restrict GPU compute scheduling and reduce throughput.
reg add "HKCU\Software\Microsoft\GameBar" /v AutoGameModeEnabled /t REG_DWORD /d 0 /f >nul
:: NOTE: HAGS OFF — continuous GPU compute workloads suffer scheduling overhead with HAGS enabled.
::       Requires reboot to take effect.
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode /t REG_DWORD /d 1 /f >nul

:: ----------------------------------------
:: Stop background services
:: ----------------------------------------

echo [3/6] Stopping background services...

net stop WSearch >nul 2>&1 && echo       WSearch stopped. || echo       WSearch already stopped or not found.
net stop SysMain >nul 2>&1 && echo       SysMain stopped. || echo       SysMain already stopped or not found.

:: ----------------------------------------
:: Kill background processes
:: ----------------------------------------

echo [4/6] Killing background processes...

for %%p in (chrome.exe msedge.exe Teams.exe OneDrive.exe SearchHost.exe) do (
    taskkill /F /IM %%p >nul 2>&1
)

:: ----------------------------------------
:: Disable Defender real-time monitoring
:: ----------------------------------------

echo [5/6] Disabling Defender real-time monitoring...

powershell -ExecutionPolicy Bypass -Command "Set-MpPreference -DisableRealtimeMonitoring $true" >nul 2>&1

:: ----------------------------------------
:: Smart App Control off (if supported)
:: ----------------------------------------

echo [6/6] Disabling Smart App Control...

:: NOTE: This registry key exists only on Windows 11 22H2+; silently skipped on older versions.
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy" /v VerifiedAndReputablePolicyState /t REG_DWORD /d 0 /f >nul 2>&1

:: ----------------------------------------

echo.
echo ========================================
echo  Preparation Done
echo ========================================
endlocal
exit /b 0
