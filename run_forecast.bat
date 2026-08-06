@echo off
cd /d "C:\Users\Pillar0713\Typhoon"
rem Messages in this file are ASCII-only on purpose: cmd emits the bytes as-is,
rem and the log is read back as cp950, so non-ASCII text would come out garbled.
rem Log rotation: keep the last 20000 lines (about 60 runs) to bound growth.
powershell -NoProfile -Command "if (Test-Path forecast_log.txt) { (Get-Content forecast_log.txt -Tail 20000) | Set-Content forecast_log.txt }"
"C:\Users\Pillar0713\AppData\Local\Programs\Python\Python313\python.exe" forecast.py >> forecast_log.txt 2>&1
call "%~dp0push_to_github.bat" >> forecast_log.txt 2>&1
