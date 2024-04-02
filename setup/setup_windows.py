import subprocess
import os
import filecmp
import logging
import shutil
import sysconfig
import setup_common
import sys

errors = 0  # Определение переменной 'errors'
log = logging.getLogger('sd')

# ANSI escape-код для желтого цвета
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'


def install_tensorflow():
    setup_common.check_repo_version()
    setup_common.check_python()

    # Обновление pip, если необходимо
    setup_common.install('--upgrade pip')

    setup_common.install_requirements('requirements_windows_tensorflow.txt', check_no_verify_flag=True)
    subprocess.Popen('start cmd /k .\webui.bat', shell=True)


def cudnn_install():
    cudnn_src = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..\cudnn_windows'
    )
    cudnn_dest = os.path.join(sysconfig.get_paths()['purelib'], 'torch', 'lib')

    log.info(f'Проверка CUDNN файлов в {cudnn_dest}...')
    if os.path.exists(cudnn_src):
        if os.path.exists(cudnn_dest):
            # check for different files
            filecmp.clear_cache()
            for file in os.listdir(cudnn_src):
                src_file = os.path.join(cudnn_src, file)
                dest_file = os.path.join(cudnn_dest, file)
                # if dest file exists, check if it's different
                if os.path.exists(dest_file):
                    if not filecmp.cmp(src_file, dest_file, shallow=False):
                        shutil.copy2(src_file, cudnn_dest)
                else:
                    shutil.copy2(src_file, cudnn_dest)
            log.info('Копирование CUDNN файлов завершено...')
        else:
            log.warning(f'Директория {cudnn_dest} не существует')
    else:
        log.error(f'Ошибка установки: "{cudnn_src}" не может быть найден.')


def main_menu():
    setup_common.clear_screen()
    while True:
        print('\n GUI установочное меню:\n')
        print('1. Установка MicroscopeAI')
        print('2. Установка cudnn файлов')
        print('2. Запуск Web-UI в браузере')
        print('3. Выход')

        choice = input('\nСделайте выбор: ')
        print('')

        if choice == '1':
            install_tensorflow()
            break
        elif choice == '2':
            cudnn_install()
        elif choice == '3':
            subprocess.Popen('start cmd /c .\webui.bat', shell=True) # /k оставить треминал открытым. /c вместо этого закрыть его
            sys.exit()
        elif choice == '4':
            print('Выход из меню')
            sys.exit()
        else:
            print('Выберите между 1-4')


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    main_menu()
