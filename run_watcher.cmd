@echo off
setlocal
cd /d "%~dp0"
start "PaperScope Watcher" /MIN python git_watcher.py
echo Watcher started in the background.
