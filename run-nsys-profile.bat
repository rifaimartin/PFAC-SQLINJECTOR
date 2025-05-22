@echo off
setlocal enabledelayedexpansion

echo ================================================== 
echo NSYS Profiling Script - SQL Injection Detection
echo ==================================================
echo.

REM Check if nsys is available
nsys --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: nsys not found! Please install NVIDIA Nsight Systems.
    echo Download from: https://developer.nvidia.com/nsight-systems
    pause
    exit /b 1
)

echo NSYS found! Starting profiling...
echo.

REM Create profiling directories
if not exist "profiling" mkdir profiling
if not exist "profiling\PFAC" mkdir profiling\PFAC
if not exist "profiling\Aho-Corasick" mkdir profiling\Aho-Corasick

REM Define arrays
set RISKS=Low Medium High Critical
set SIZES=10 100 1000 5000 10000

echo ==================================================
echo Profiling PFAC Algorithm...
echo ==================================================

for %%r in (%RISKS%) do (
    for %%s in (%SIZES%) do (
        set dataset=sql_dataset_%%r_%%s
        set profile_name=PFAC-%%r-%%s
        echo Profiling: !profile_name!
        
        if exist "!dataset!.csv" (
            echo Command: nsys profile -o profiling\PFAC\!profile_name! --stats=true PFAC.exe -d !dataset!
            nsys profile -o "profiling\PFAC\!profile_name!" --stats=true PFAC.exe -d "!dataset!"
            
            if exist "profiling\PFAC\!profile_name!.nsys-rep" (
                echo ✓ Profile saved: !profile_name!.nsys-rep
            ) else (
                echo ✗ Profile failed: !profile_name!
            )
            echo ---
        ) else (
            echo Dataset !dataset!.csv not found, skipping...
        )
    )
)

echo.
echo ==================================================
echo Profiling Aho-Corasick Algorithm...
echo ==================================================

for %%r in (%RISKS%) do (
    for %%s in (%SIZES%) do (
        set dataset=sql_dataset_%%r_%%s
        set profile_name=Aho-Corasick-%%r-%%s
        echo Profiling: !profile_name!
        
        if exist "!dataset!.csv" (
            echo Command: nsys profile -o profiling\Aho-Corasick\!profile_name! --stats=true Aho-Corasick.exe -d !dataset!
            nsys profile -o "profiling\Aho-Corasick\!profile_name!" --stats=true Aho-Corasick.exe -d "!dataset!"
            
            if exist "profiling\Aho-Corasick\!profile_name!.nsys-rep" (
                echo ✓ Profile saved: !profile_name!.nsys-rep
            ) else (
                echo ✗ Profile failed: !profile_name!
            )
            echo ---
        ) else (
            echo Dataset !dataset!.csv not found, skipping...
        )
    )
)

echo.
echo ==================================================
echo Profiling Summary
echo ==================================================

echo PFAC Profiles Generated:
for /f %%f in ('dir /b "profiling\PFAC\*.nsys-rep" 2^>nul') do (
    echo   ✓ %%f
)

echo.
echo Aho-Corasick Profiles Generated:
for /f %%f in ('dir /b "profiling\Aho-Corasick\*.nsys-rep" 2^>nul') do (
    echo   ✓ %%f
)

echo.
echo ==================================================
echo Profiling completed!
echo.
echo To view profiles, use:
echo   nsys-ui profiling\PFAC\[profile-name].nsys-rep
echo   nsys-ui profiling\Aho-Corasick\[profile-name].nsys-rep
echo.
echo Or open with Nsight Systems GUI application.
echo ==================================================
pause