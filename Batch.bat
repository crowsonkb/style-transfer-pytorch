
@echo off
echo Style transfer Quick CLI tool

setlocal enabledelayedexpansion enableextensions
set /p Input="Input Path:"
echo Style Path(I would advise using less than 10 style images, i'm not sure how many the AI can input)
set /p Style="(Style folder path):"
set STYLES=
for %%x in (%Style%\*) do set LIST=!LIST! %%x
set LIST=%LIST:~1%
echo %LIST%
set /p Output="Output Path:"

style_transfer -h
set /p Additionals="Additional Settings, Use Default if you're not sure what to do:" || set Additionals=--web --host localhost --browser
  ::--web --host localhost  -ms 64 -s 128 -cw 0.05 -ss 0.1  --browser -i 500 -ii 1000 --save-every 75 -tw 3 (my favorite)

cd /d %Input%
set InputShort=%Input:~-15%
set ListShort=%LIST:~-15%
set OutputShort=%Output:~-15%
set AddShort=%Additionals:~-15%
echo "Input"       %InputShort%
echo "Style(s)"    %ListShort%
echo "Output"      %OutputShort%
echo "Additionals" %AddShort%


for %%i in (*) do (
echo %%i
style_transfer %Input%\%%i %LIST% -o %Output%\%%i %Additionals%%Browser%
)
:END
echo Done!
PING -n 4 localhost>nul
