@echo off
echo ========================================
echo Thermal Image Enhancement Application
echo ========================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Generating sample data...
python src/utils/sample_generator.py

echo.
echo Starting the application...
python main.py

pause 