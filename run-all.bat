@echo off
setlocal enabledelayedexpansion

echo ================================================== 
echo SQL Injection Detection - Automated Test Runner
echo ==================================================
echo.

REM Check if executables exist
if not exist "PFAC.exe" (
    echo ERROR: PFAC.exe not found!
    pause
    exit /b 1
)

if not exist "Aho-Corasick.exe" (
    echo ERROR: Aho-Corasick.exe not found!
    pause
    exit /b 1
)

echo Found both executables!
echo.

REM Create results directory
if not exist "results" mkdir results
if not exist "results\PFAC" mkdir results\PFAC
if not exist "results\Aho-Corasick" mkdir results\Aho-Corasick

REM Define arrays (using space-separated values)
set RISKS=Low Medium High Critical
set SIZES=10 100 1000 5000 10000

echo ==================================================
echo Running PFAC Tests...
echo ==================================================

REM Test PFAC
for %%r in (%RISKS%) do (
    for %%s in (%SIZES%) do (
        set dataset=sql_dataset_%%r_%%s
        echo Testing PFAC with !dataset!...
        
        if exist "!dataset!.csv" (
            PFAC.exe -d "!dataset!"
            if exist "result_PFAC_!dataset!.txt" (
                move "result_PFAC_!dataset!.txt" "results\PFAC\"
            )
            echo ---
        ) else (
            echo Dataset !dataset!.csv not found, skipping...
        )
    )
)

echo.
echo ==================================================
echo Running Aho-Corasick Tests...
echo ==================================================

REM Test Aho-Corasick
for %%r in (%RISKS%) do (
    for %%s in (%SIZES%) do (
        set dataset=sql_dataset_%%r_%%s
        echo Testing Aho-Corasick with !dataset!...
        
        if exist "!dataset!.csv" (
            Aho-Corasick.exe -d "!dataset!"
            
            REM Check for different possible output file names
            if exist "result_Aho-Corasick_!dataset!.txt" (
                move "result_Aho-Corasick_!dataset!.txt" "results\Aho-Corasick\"
                echo Moved: result_Aho-Corasick_!dataset!.txt
            ) else if exist "result_AhoCorasick_!dataset!.txt" (
                move "result_AhoCorasick_!dataset!.txt" "results\Aho-Corasick\"
                echo Moved: result_AhoCorasick_!dataset!.txt
            ) else if exist "result_AC_!dataset!.txt" (
                move "result_AC_!dataset!.txt" "results\Aho-Corasick\"
                echo Moved: result_AC_!dataset!.txt
            ) else (
                echo Warning: No result file found for !dataset!
                echo Looking for files matching pattern...
                dir result_*!dataset!*.txt 2>nul
            )
            echo ---
        ) else (
            echo Dataset !dataset!.csv not found, skipping...
        )
    )
)

echo.
echo ==================================================
echo All tests completed!
echo Results saved in: results\
echo ==================================================
pause