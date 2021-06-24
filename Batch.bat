@echo off


echo Style transfer Quick CLI tool
setlocal enabledelayedexpansion enableextensions

set /p Input="Input Path:"
cd /d %Input%
echo.
echo Style Path(I would advise using less than 10 style images, i'm not sure how many the AI can input)
set /p Style="(Style folder path):"
set STYLES=
for %%x in (%Style%\*) do set LIST=!LIST! %%x
set LIST=%LIST:~1%
echo %LIST%
echo.
set /p Output="Output Path:"
echo.
style_transfer -h
echo "Additional Settings, Use Default if you're not sure what to do"
echo.
set /p Additionals="Additionals:" || set Additionals=--web --browser
  ::--web --host localhost --browser   -ms 128 -s 1024 -cw 0.04 -ss 0.05   -i 500 -ii 1000 --save-every 75 (my favorite)
echo.
set InputShort=%Input:~-15%
set ListShort=%LIST:~-15%
set OutputShort=%Output:~-15%
set AddShort=%Additionals:~-15%
echo "Input"       ...%InputShort%
echo "Style(s)"    ...%ListShort%
echo "Output"      ...%OutputShort%
echo "Additionals" ...%AddShort%
echo.
for %%i in (*) do (
echo %%i
style_transfer %Input%\%%i %LIST% -o %Output%\%%i %Additionals%%Browser%
)
:END
echo Done!
PING -n 6 localhost>nul
