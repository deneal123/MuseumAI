@echo off

:: Деактивация активной среды
call .\venv\Scripts\deactivate.bat

:: Проверка на локальные модули
python .\setup\check_local_modules.py --no_question

:: Активация виртуальной среды
call .\venv\Scripts\activate.bat
set PATH=%PATH%;%~dp0venv\Lib\site-packages\torch\lib

:: Валидация requirements
python.exe .\setup\validate_requirements.py

:: Установка стандартных настроек конфигурации
python.exe config_file.py

:: Очистка setup.log
python.exe clear_setup_log.py

:: Запускает webui.py скрипт c веб интерфейсом.
if %errorlevel% equ 0 (
    REM Был ли запущен батник двойным кликом?
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo Этот скрипт был запущен двойным кликом.
	cmd /k python.exe -m streamlit run webui_demo.py %*
    ) ELSE (
        REM echo Этот скрипт был запущен с помощью командной строки.
        python.exe -m streamlit run webui_demo.py %*
    )
)
