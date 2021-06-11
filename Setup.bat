@echo off

pip install -e %~pd0
echo Done!
PING -n 4 localhost>nul
