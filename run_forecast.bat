@echo off
cd /d "C:\Users\Pillar0713\Typhoon"
rem 日誌輪替：只保留最後 5000 行，避免無限增長
powershell -command "if (Test-Path forecast_log.txt) { (Get-Content forecast_log.txt -Tail 5000) | Set-Content forecast_log.txt }"
"C:\Users\Pillar0713\AppData\Local\Programs\Python\Python313\python.exe" forecast.py >> forecast_log.txt 2>&1
call "%~dp0push_to_github.bat" >> forecast_log.txt 2>&1
