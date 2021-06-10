@echo off

pip install -e %CD:~0,3%%~p0
echo Done!
PING -n 4 localhost>nul