@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

if "%SPAWNN_HOME%"=="" goto guessspawnnhome
goto javahome

:guessspawnnhome
set SPAWNN_BATCHDIR=%~dp0
set SPAWNN_HOME=%SPAWNN_BATCHDIR%
echo SPAWNN_HOME is not set. Trying the directory '%SPAWNN_HOME%'
goto javahome

rem ############################
rem ###                      ###
rem ###  Searching for Java  ###
rem ###                      ###
rem ############################

:javahome
set LOCAL_JRE_JAVA=%SPAWNN_HOME%\jre\bin\java.exe
if exist "%LOCAL_JRE_JAVA%" goto localjre
goto checkjavahome

:localjre
set JAVA=%LOCAL_JRE_JAVA%
echo Using local jre: %JAVA%...
goto start

:checkjavahome
if "%JAVA_HOME%"=="" goto checkpath
set JAVA_CHECK=%JAVA_HOME%\bin\java.exe
if exist "%JAVA_CHECK%" goto globaljre 
goto error3
rem goto globaljre

:globaljre
set JAVA=%JAVA_HOME%\bin\java
echo Using global jre: %JAVA%...
goto start

:checkpath
java -version 2> nul:
if errorlevel 1 goto error2
goto globaljrepath

:globaljrepath
set JAVA=java
echo Using global jre found on path: %JAVA%
goto start


rem #############################
rem ###                       ###
rem ###  Starting Spawnn      ###
rem ###                       ###
rem #############################

:start
set SPAWNN_LIBRARIES=
for %%f in (%SPAWNN_HOME%\lib\*.jar) do (set SPAWNN_LIBRARIES=!SPAWNN_LIBRARIES!;%%f)
set COMPLETE_CLASSPATH=%SPAWNN_LIBRARIES%
echo Starting Spawnn from '%SPAWNN_HOME%'
rem echo The complete classpath is '%COMPLETE_CLASSPATH%'

"%JAVA%" -classpath "%COMPLETE_CLASSPATH%" -Xmx1g spawnn.gui.SpawnnGui

goto end

rem ########################
rem ###                  ###
rem ###  Error messages  ###
rem ###                  ###
rem ########################

:error2
echo.
echo ERROR: Java cannot be found. 
echo Please install Java properly (check if JAVA_HOME is 
echo correctly set or ensure that 'java' is part of the 
echo PATH environment variable).
echo.
pause
goto end

:error3
echo.
echo ERROR: Java cannot be found in the path JAVA_HOME
echo Please install Java properly (it seems that the 
echo environment variable JAVA_HOME does not point to 
echo a Java installation).
echo.
pause
goto end

rem #############
rem ###       ###
rem ###  END  ###
rem ###       ###
rem #############

:end
