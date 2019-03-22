forfiles /S /M *.py /C "cmd /c 2to3 -w @file"

PAUSE