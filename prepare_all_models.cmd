@echo off
setlocal
cd /d "%~dp0"

call :pick_python
echo [INFO] Using Python: "%PYTHON_EXE%"

set "READY_MODELS="
set "FAILED_MODELS="

call :prepare_model baseline baseline_fraud_checkpoint.pkl
call :prepare_model phobert phobert_fraud_checkpoint.pt tests\fixtures\mini_samples.jsonl
call :prepare_model mfinbert mfinbert_fraud_checkpoint.pt tests\fixtures\mini_samples.jsonl

echo.
if defined READY_MODELS (
  echo [DONE] Ready models: %READY_MODELS%
)
if defined FAILED_MODELS (
  echo [WARN] Could not prepare: %FAILED_MODELS%
  echo [WARN] The demo server can still start; missing models will be disabled in the UI.
)
exit /b 0

:prepare_model
set "MODEL_NAME=%~1"
set "CHECKPOINT_FILE=%~2"
set "DATASET_PATH=%~3"

if exist "%CHECKPOINT_FILE%" (
  echo [SKIP] %MODEL_NAME% already has checkpoint "%CHECKPOINT_FILE%".
  call set "READY_MODELS=%%READY_MODELS%% %MODEL_NAME%"
  exit /b 0
)

echo.
echo [RUN ] Preparing checkpoint for %MODEL_NAME% ...
if defined DATASET_PATH (
  echo [INFO] Using dataset "%DATASET_PATH%" for fast demo preparation.
  "%PYTHON_EXE%" main.py --prepare-api --model %MODEL_NAME% --dataset "%DATASET_PATH%"
) else (
  "%PYTHON_EXE%" main.py --prepare-api --model %MODEL_NAME%
)
if errorlevel 1 (
  echo [FAIL] Could not prepare %MODEL_NAME%.
  call set "FAILED_MODELS=%%FAILED_MODELS%% %MODEL_NAME%"
  exit /b 0
)
echo [ OK ] %MODEL_NAME% checkpoint created.
call set "READY_MODELS=%%READY_MODELS%% %MODEL_NAME%"
exit /b 0

:pick_python
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
  set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
  exit /b 0
)

if exist "C:\Python314\python.exe" (
  set "PYTHON_EXE=C:\Python314\python.exe"
  exit /b 0
)

set "PYTHON_EXE=python"
exit /b 0
