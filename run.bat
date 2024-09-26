@echo off
REM Check for correct number of arguments
if "%~3"=="" (
    echo Usage: %0 path_to_context.json path_to_test.json path_to_output_prediction.csv
    exit /b 1
)

REM Assign arguments to variables
set CONTEXT_PATH=%1
set TEST_PATH=%2
set OUTPUT_PATH=%3

REM Paths to models
set PARAGRAPH_MODEL_DIR=paragraph_selection_model
set SPAN_MODEL_DIR=span_prediction_model

REM Step 1: Train the paragraph selection model
echo Training paragraph selection model...
python train_multi_select.py --output_dir "%PARAGRAPH_MODEL_DIR%"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM Step 2: Train the span prediction model
echo Training span prediction model...
python train_span_selection.py --output_dir "%SPAN_MODEL_DIR%"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM Step 3: Run inference
echo Running inference...
python inference.py "%CONTEXT_PATH%" "%TEST_PATH%" "%OUTPUT_PATH%"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo Inference complete. Predictions saved to "%OUTPUT_PATH%"