@echo off
setlocal
cd /d "%~dp0"

echo FS_MS focused cardiac and microstate pipeline
echo.
echo The run is unattended except for one terminal prompt after the global K=6 fit.
echo At that prompt, inspect the opened topography image, enter six labels in State 1..6 order,
echo confirm them, and leave this terminal open. The pipeline will then continue automatically.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ace_manual_pipeline.ps1"
set "FSMS_EXIT=%ERRORLEVEL%"

echo.
if not "%FSMS_EXIT%"=="0" (
    echo FS_MS pipeline FAILED with exit code %FSMS_EXIT%.
    echo Review pipeline_run.log and pipeline_audit.json in focused_analysis_20260726.
) else (
    echo FS_MS pipeline completed successfully.
)

exit /b %FSMS_EXIT%
