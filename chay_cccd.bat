@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  python cccd_scanner.py
) else (
  python cccd_scanner.py %*
)
echo.
echo Ket qua nam trong thu muc "output". Bam phim bat ky de dong.
pause >nul
