@echo off
cd /d "C:\Users\Pillar0713\Typhoon"

git add docs/
git add forecast.py generate_forecast_website.py

git diff --cached --quiet
if %errorlevel% == 0 (
    echo [INFO] 無變更，跳過推送
    exit /b 0
)

rem 取真正的 UTC 時間（Get-Date 預設為本地時間，不可直接標成 UTC）
for /f "usebackq tokens=*" %%i in (`powershell -command "(Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')"`) do set TIMESTAMP=%%i
git commit -m "Auto update: %TIMESTAMP% UTC"

git pull --rebase --autostash origin main
if %errorlevel% neq 0 (
    echo [ERROR] git pull --rebase 失敗，請手動處理衝突
    exit /b 1
)

git push origin main
if %errorlevel% == 0 (
    echo [INFO] 成功推送至 GitHub
) else (
    echo [WARN] 第一次推送失敗，30秒後重試...
    timeout /t 30 /nobreak >nul
    git push origin main
    if %errorlevel% == 0 (
        echo [INFO] 重試推送成功
    ) else (
        echo [ERROR] 推送失敗，請檢查 forecast_log.txt
    )
)
