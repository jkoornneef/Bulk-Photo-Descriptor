@echo off
REM =========================================================================
REM processfaces.bat - build a face database (pkl) from a folder of photos.
REM
REM Usage:
REM   processfaces.bat [known_dir] [face_db]
REM
REM   known_dir  Folder containing one subfolder per person:
REM                known_dir\Chris\*.jpg
REM                known_dir\Jim\*.jpg
REM              Default: .\faces
REM
REM   face_db    Where to write the database.
REM              Default: <known_dir name>.pkl  (so .\faces -> faces.pkl)
REM              Pass any name to keep several databases side by side.
REM
REM Examples:
REM   processfaces.bat                         REM .\faces -> faces.pkl
REM   processfaces.bat my_friends              REM my_friends -> my_friends.pkl
REM   processfaces.bat my_friends friends.pkl  REM custom folder and output
REM =========================================================================

setlocal

set "KNOWN_DIR=%~1"
if "%KNOWN_DIR%"=="" set "KNOWN_DIR=.\faces"

if not "%~2"=="" (
    set "FACE_DB=%~2"
) else (
    for %%F in ("%KNOWN_DIR%") do set "FACE_DB=%%~nF.pkl"
)

python describe_images.py --enroll "%KNOWN_DIR%" --face-db "%FACE_DB%"
