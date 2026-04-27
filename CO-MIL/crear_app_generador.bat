@echo off
setlocal

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%\.."

if not exist "env\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en env\Scripts\python.exe
    echo Crea o activa el entorno antes de generar la app.
    exit /b 1
)

env\Scripts\python.exe -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller no esta instalado en el entorno virtual.
    echo Instala primero con:
    echo     env\Scripts\python.exe -m pip install pyinstaller
    exit /b 1
)

echo.
echo [Co-MIL] Generando app de etiquetado...
env\Scripts\python.exe -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --name CoMIL_Etiquetador ^
    CO-MIL\generador_bolsas.py

if errorlevel 1 (
    echo.
    echo [ERROR] La generacion de la app fallo.
    exit /b 1
)

echo.
echo [OK] App generada en:
echo     dist\CoMIL_Etiquetador\
echo.
echo Comparte toda esa carpeta o comprimela en .zip antes de enviarla.
exit /b 0
