@echo off
setlocal
cd /d "%~dp0"

call prepare_all_models.cmd

call start_demo_server.cmd
exit /b %errorlevel%
