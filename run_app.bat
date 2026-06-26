@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   时间周期择时监测平台  启动中...
echo   浏览器将自动打开 http://localhost:8501
echo   关闭本窗口即可停止平台
echo ============================================
set PYTHONUTF8=1
python -m streamlit run app.py
pause
