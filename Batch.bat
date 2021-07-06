@echo off

echo Style transfer Quick CLI batch tool
setlocal enabledelayedexpansion enableextensions

set /p Input="Input Path:"
cd /d %Input%
set /p Temp="Backup Path:"
echo %Temp%
echo.
echo Style Path(I would advise using less than 10 style images, i'm not sure how many the AI can input)
set /p Style="(Style folder path):"
set STYLES=
for %%x in (%Style%\*) do set LIST=!LIST! %%x
set LIST=%LIST:~1%
echo %LIST%
echo.
set /p Output="Output Path:"
cls

style_transfer -h
echo "Additional Settings, Use Default if you're not sure what to do"
echo.
set Browser=--browser
set /p Additionals="Additionals:" || set Additionals=--web --host localhost
  ::--web --host localhost -ms 128 -s 1280 -cw 0.05 -i 128 -ii 2500 --save-every 75 -r 1(my favorite)
echo.
cls

set InputShort=%Input:~-25%
set ListShort=%LIST:~-25%
set OutputShort=%Output:~-25%
set AddShort=%Additionals:~-25%
set BackShort=%Temp:~-25% 
set Browser=--browser
echo "Input"       ...%InputShort%
echo "Backup"      ...%BackShort%
echo "Style(s)"    ...%ListShort%
echo "Output"      ...%OutputShort%
echo "Additionals" ...%AddShort%

for %%i in (*) do (
echo %input%\%%i
style_transfer %Input%\%%i %LIST% -o %Output%\%%i %Additionals% %Browser%
move /Y "%Input%\%%i" %Temp%
goto skip
cls
)

:skip 
for %%i in (*) do (
echo %input%\%%i
style_transfer %Input%\%%i %LIST% -o %Output%\%%i %Additionals%
move /Y "%Input%\%%i" %Temp%
cls
)
:END
echo Done
PING -n 11 localhost>nul
