@echo off
setlocal
cd /d "%~dp0"

call :pick_python
echo [INFO] Using Python: "%PYTHON_EXE%"

set "FRAUD_API_ALLOW_TRAINING=0"
set "FRAUD_API_PRELOAD_MODELS=baseline,phobert,mfinbert"
call :configure_ocr_tools

echo [INFO] Starting demo server on http://127.0.0.1:8000/
echo [INFO] Open the root page and drop a .txt or .pdf file into the demo UI.
"%PYTHON_EXE%" -m uvicorn app:app --host 127.0.0.1 --port 8000
exit /b %errorlevel%

:configure_ocr_tools
if exist "C:\Windows\System32" (
  set "PATH=C:\Windows\System32;%PATH%"
)

if exist "C:\poppler\Library\bin\pdfinfo.exe" (
  set "POPPLER_PATH=C:\poppler\Library\bin"
  set "PATH=C:\poppler\Library\bin;%PATH%"
  echo [INFO] Poppler detected at "%POPPLER_PATH%"
) else if exist "C:\Program Files\poppler\Library\bin\pdfinfo.exe" (
  set "POPPLER_PATH=C:\Program Files\poppler\Library\bin"
  set "PATH=C:\Program Files\poppler\Library\bin;%PATH%"
  echo [INFO] Poppler detected at "%POPPLER_PATH%"
) else if exist "C:\Program Files (x86)\poppler\Library\bin\pdfinfo.exe" (
  set "POPPLER_PATH=C:\Program Files (x86)\poppler\Library\bin"
  set "PATH=C:\Program Files (x86)\poppler\Library\bin;%PATH%"
  echo [INFO] Poppler detected at "%POPPLER_PATH%"
)

if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
  set "TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
  set "PATH=C:\Program Files\Tesseract-OCR;%PATH%"
  echo [INFO] Tesseract detected at "%TESSERACT_CMD%"
) else if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" (
  set "TESSERACT_CMD=C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
  set "PATH=C:\Program Files (x86)\Tesseract-OCR;%PATH%"
  echo [INFO] Tesseract detected at "%TESSERACT_CMD%"
) else if exist "%LocalAppData%\Programs\Tesseract-OCR\tesseract.exe" (
  set "TESSERACT_CMD=%LocalAppData%\Programs\Tesseract-OCR\tesseract.exe"
  set "PATH=%LocalAppData%\Programs\Tesseract-OCR;%PATH%"
  echo [INFO] Tesseract detected at "%TESSERACT_CMD%"
)
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
