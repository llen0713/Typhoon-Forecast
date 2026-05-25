@echo off
cd /d "C:\Users\Pillar0713\Typhoon"
"C:\Users\Pillar0713\AppData\Local\Programs\Python\Python313\python.exe" forecast.py >> forecast_log.txt 2>&1
call push_to_github.bat >> forecast_log.txt 2>&1
