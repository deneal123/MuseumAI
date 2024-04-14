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


def sync_bits_and_bytes_files():
    import filecmp

    """
    Проверяйте файлы на наличие «разных» битов и байтов и копируйте только при необходимости.
    Эта функция специфична для ОС Windows.
    """

    # Выполнять только в Windows
    if os.name != 'nt':
        print('Эта функция применима только к ОС Windows...')
        return

    try:
        log.info(f'Копирование bitsandbytes файлов...')
        # Определить исходный и конечный каталоги
        source_dir = os.path.join(os.getcwd(), 'bitsandbytes_windows')

        dest_dir_base = os.path.join(
            sysconfig.get_paths()['purelib'], 'bitsandbytes'
        )

        # Очистить кеш сравнения файлов
        filecmp.clear_cache()

        # Перебрать каждый файл в исходном каталоге
        for file in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, file)

            # Определите каталог назначения на основе имени файла
            if file in ('main.py', 'paths.py'):
                dest_dir = os.path.join(dest_dir_base, 'cuda_setup')
            else:
                dest_dir = dest_dir_base

            dest_file_path = os.path.join(dest_dir, file)

            # Сравните исходный файл с файлом назначения
            if os.path.exists(dest_file_path) and filecmp.cmp(
                source_file_path, dest_file_path
            ):
                log.debug(
                    f'Пропуск {source_file_path}, так как он уже существует в {dest_dir}'
                )
            else:
                # Скопируйте файл из источника в место назначения, сохранив метаданные исходного файла.
                log.debug(f'Копирование {source_file_path} в {dest_dir}')
                shutil.copy2(source_file_path, dest_dir)

    except FileNotFoundError as fnf_error:
        log.error(f'Ошибка файл не найден: {fnf_error}')
    except PermissionError as perm_error:
        log.error(f'Ошибка доступа: {perm_error}')
    except Exception as e:
        log.error(f'Непредвиденная ошибка: {e}')


def install_torch1():
    setup_common.check_repo_version()
    setup_common.check_python()

    # Обновление pip, если необходимо
    setup_common.install('--upgrade pip')

    if setup_common.check_torch() == 2:
        input(
            f'{YELLOW}\nTorch 2 уже установлен в среду venv. Чтобы установить torch 1 удалите venv и перезапустите setup.bat\n\nНажмите enter, чтобы продолжить...{RESET_COLOR}'
        )
        return

    # setup_common.install(
    #     'torch==1.12.1+cu116 torchvision==0.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116',
    #     'torch torchvision'
    # )
    # setup_common.install(
    #     'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl -U -I --no-deps',
    #     'xformers-0.0.14'
    # )
    setup_common.install_requirements('requirements_windows_torch1.txt', check_no_verify_flag=False)
    # setup_common.configure_accelerate(run_accelerate=True)
    # run_cmd(f'accelerate config')


def install_torch2():
    setup_common.check_repo_version()
    setup_common.check_python()

    # Обновление pip, если необходимо
    setup_common.install('--upgrade pip')

    if setup_common.check_torch() == 1:
        input(
            f'{YELLOW}\nTorch 1 уже установлен в среду venv. Чтобы установить torch 1 удалите среду venv и перезапустите setup.bat\n\nНажмите любую кнопку...{RESET_COLOR}'
        )
        return

    # setup_common.install(
    #     'torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118',
    #     'torch torchvision'
    # )
    setup_common.install_requirements('requirements_windows_torch2.txt', check_no_verify_flag=False)
    # install('https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl', 'triton', reinstall=reinstall)
    # setup_common.configure_accelerate(run_accelerate=True)
    # run_cmd(f'accelerate config')



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
        print('1. Установка MuseumAI')
        print('2. Установка cudnn файлов')
        print('3. Установка bitsandbytes-windows')
        print('4. Запуск Web-UI в браузере')
        print('5. Выход')

        choice = input('\nСделайте выбор: ')
        print('')

        if choice == '1':
            while True:
                print('1. Torch 1')
                print('2. Torch 2 (Рекомендовано)')
                print('3. Отмена')
                choice_torch = input('\nСделайте выбор: ')
                print('')

                if choice_torch == '1':
                    install_torch1()
                    break
                elif choice_torch == '2':
                    install_torch2()
                    break
                elif choice_torch == '3':
                    break
                else:
                    print('Выберите между 1-3.')
        elif choice == '2':
            cudnn_install()
        elif choice == '3':
            setup_common.install('--upgrade bitsandbytes-windows', reinstall=True)
        elif choice == '4':
            subprocess.Popen('start cmd /c .\webui.bat', shell=True) # /k оставить треминал открытым. /c вместо этого закрыть его
            sys.exit()
        elif choice == '5':
            print('Выход из меню')
            sys.exit()
        else:
            print('Выберите между 1-5')


if __name__ == '__main__':
    setup_common.ensure_base_requirements()
    setup_common.setup_logging()
    main_menu()
