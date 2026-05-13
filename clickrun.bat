@echo off
call venv\Scripts\activate.bat
omnivoice-demo --ip 0.0.0.0 --port 8001
pause