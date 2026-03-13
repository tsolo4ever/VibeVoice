@echo off
cd /d "%~dp0"
python app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Failed to start. Check that:
    echo   1. Python is on PATH
    echo   2. VibeVoice is installed: pip install -e D:\GitHub\VibeVoice
    echo   3. EPUB deps installed:    pip install ebooklib beautifulsoup4 lxml
    echo   4. ffmpeg is on PATH:      https://ffmpeg.org/download.html
    echo.
    pause
)
