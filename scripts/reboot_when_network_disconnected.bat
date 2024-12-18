@echo off
set ping_ip=dg2raptorlake.ikor.intel.com
set failure_count=0
set ok_count=4
set timeout_secs=30
set connection_ok_count=0
set connection_error_count=0
set max_connection_error_count=3


:start
:: calling the ping function
call :connection_test

:: Processing the network "up" state
if "%network_state%"=="up" (
    echo INFO: You have an active connection.
    set connection_error_count=0
    set /a connection_ok_count+=1
) else (
    set /a connection_error_count+=1
    set connection_ok_count=0
)

:: Processing the network "down" state
if "%network_state%"=="down" (
    if %connection_error_count% geq %max_connection_error_count% (
        echo ERROR: You do not have an active connection.
        goto poweroff
    ) else (
        echo INFO: FAILURE: That failed [%connection_error_count%] times, NOT good. lets try again... 
        goto start
    )
)


:: Check the ping_count against the failure_count
if "%ok_count%" leq "%connection_ok_count%" (
    echo INFO: SUCCESS: No issue
    goto :end
)


timeout /t %timeout_secs%
goto start



:: connection_test function
goto skip_connection_test
:connection_test
:: Getting the successful ping count
echo INFO: Checking connection, please hang tight for a second...
for /f "tokens=5 delims==, " %%p in ('ping -n 4 %ping_ip% ^| findstr /i "Received"') do set ping_count=%%p; echo %ping_count%


:: Check the ping_count against the failure_count
if "%ping_count%" leq "%failure_count%" (
    set network_state=down
) else (
    set network_state=up
)
goto :eof
:skip_connection_test




:: Power off 
:poweroff
echo INFO: Restarting PC in 5 seconds.  Press any key to abort.
shutdown -r -t 5 -f
pause > nul
shutdown -a
goto end

:end 

