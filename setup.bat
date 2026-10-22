@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Cai dat va Khoi dong Du an AI Tuyen dung

cd /d "%~dp0"

echo ===================================================
echo   CAI DAT VA KHOI DONG DU AN AI DANH GIA UNG VIEN
echo ===================================================
echo.

:: 1. Kiem tra requirements.txt
if not exist "requirements.txt" (
    echo [LOI] Khong tim thay file requirements.txt trong thu muc hien tai!
    goto ERROR_EXIT
)

:: 2. Kiem tra Python
echo [1/4] Kiem tra moi truong Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Python chua duoc cai dat hoac chua duoc pride vao PATH.
    goto ERROR_EXIT
)
python --version

:: 3. Tao moi truong ao (venv)
echo.
echo [2/4] Kiem tra / Tao moi truong ao (venv)...
if not exist "venv\Scripts\python.exe" (
    echo Dang tao venv moi...
    python -m venv venv
    if errorlevel 1 (
        echo [LOI] Khong the tao venv.
        goto ERROR_EXIT
    )
    echo [OK] Da tao venv thanh cong.
) else (
    echo [OK] Moi truong ao venv da ton tai.
)

:: 4. Kich hoat venv va cai dat thu vien
echo.
echo [3/4] Cap nhat pip va cai dat thu vien...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [LOI] Khong the nang cap pip.
    goto ERROR_EXIT
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [LOI] Cai dat thu vien tu requirements.txt that bai.
    goto ERROR_EXIT
)

:: 5. Chay kiem thu (pytest)
echo.
echo [4/4] Chay kiem thu voi pytest...
python -m pytest -v
if errorlevel 1 (
    echo.
    echo [CANH BAO] Kiem thu thieu hoac test case thiet bai!
    echo Ngung khoi dong ung dung de dam bao an toan.
    goto ERROR_EXIT
)

:: 6. Kiem tra app.py va khoi dong Streamlit
echo.
if not exist "app.py" (
    echo [LOI] Khong tim thay file app.py de chay Streamlit!
    goto ERROR_EXIT
)

echo ===================================================
echo   KHOI DONG UNG DUNG STREAMLIT...
echo ===================================================
echo.

python -m streamlit run app.py
goto END

:ERROR_EXIT
echo.
echo ===================================================
echo   QUA TRINH XUY LY BI LOI! VUI LONG KIEM TRA LAI.
echo ===================================================
pause
exit /b 1

:END
pause
