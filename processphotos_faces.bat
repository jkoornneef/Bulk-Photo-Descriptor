@echo off
REM =========================================================================
REM processphotos_faces.bat - like processphotos.bat, but names enrolled
REM faces in the generated descriptions.
REM
REM Usage:
REM   processphotos_faces.bat <photos_dir> [face_db ...]
REM
REM   photos_dir  Directory of photographs to describe (required).
REM   face_db     One or more face database .pkl files to load.
REM               Pass several to merge them (e.g. faces.pkl plus a
REM               temporary cast for one event).
REM               Default: none given -> faces.pkl (describe_images.py default),
REM               written by processfaces.bat with no arguments.
REM
REM Examples:
REM   processphotos_faces.bat "D:\Photos"                          REM faces.pkl
REM   processphotos_faces.bat "D:\Photos\school play" friends.pkl  REM override
REM   processphotos_faces.bat "D:\Photos\school play" faces.pkl friends.pkl  REM merge
REM
REM Model defaults to gemma4:e4b because it reliably follows the
REM face-name instructions (llava/qwen2.5vl usually ignore them).
REM =========================================================================

setlocal enabledelayedexpansion

set "PHOTOS=%~1"
if "%PHOTOS%"=="" (
    echo Usage: %~nx0 ^<photos_dir^> [face_db ...]
    echo.
    echo Names enrolled faces in the descriptions. Default face DB: faces.pkl
    echo Pass one face_db to replace it, or several to merge them.
    exit /b 1
)
shift

set "FACE_ARGS="
:collect
if "%~1"=="" goto run
set FACE_ARGS=!FACE_ARGS! --face-db "%~1"
shift
goto collect

:run

python describe_images.py "%PHOTOS%" --model gemma4:e4b !FACE_ARGS!
