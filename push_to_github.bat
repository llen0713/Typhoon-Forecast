@echo off
rem Delayed expansion is required: %errorlevel% inside an if/else block is
rem substituted when the whole block is parsed, so the retry branch below
rem would otherwise test the FIRST push's exit code, never the retry's.
setlocal enabledelayedexpansion
cd /d "C:\Users\Pillar0713\Typhoon"

rem Messages here are ASCII-only on purpose: cmd emits the bytes as-is and the
rem log is read back as cp950, so non-ASCII text would come out garbled.

rem A Ctrl+C during a previous run can leave the repo stopped mid-rebase, which
rem wedges every later push. Abort it so this run can proceed instead of the
rem schedule failing forever.
if exist ".git\rebase-merge" (
    echo [WARN] Unfinished rebase found, running git rebase --abort
    git rebase --abort
)
if exist ".git\rebase-apply" (
    echo [WARN] Unfinished rebase found, running git rebase --abort
    git rebase --abort
)

git add docs/
git add forecast.py generate_forecast_website.py

git diff --cached --quiet
if !errorlevel! == 0 (
    echo [INFO] No staged changes, skipping push
    exit /b 0
)

rem Real UTC time: Get-Date returns local time and must not be labelled UTC
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')"`) do set TIMESTAMP=%%i
git commit -m "Auto update: !TIMESTAMP! UTC"

rem --autostash only stashes tracked changes. An UNTRACKED file that also
rem exists on the remote will still abort the rebase, so never leave a local
rem untracked copy of a file that was pushed by other means (e.g. gh api).
git pull --rebase --autostash origin main
if !errorlevel! neq 0 (
    echo [ERROR] git pull --rebase failed, resolve manually
    exit /b 1
)

git push origin main
if !errorlevel! == 0 (
    echo [INFO] Pushed to GitHub successfully
) else (
    echo [WARN] First push failed, retrying in 30s...
    timeout /t 30 /nobreak >nul
    git push origin main
    if !errorlevel! == 0 (
        echo [INFO] Retry push succeeded
    ) else (
        echo [ERROR] Push failed, check forecast_log.txt
        exit /b 1
    )
)
