import os
import re
import sys
import shutil
import argparse
import setup_common

# Получить абсолютный путь к каталогу текущего файла (каталог проекта cv_semantic_segmentation)
project_directory = os.path.dirname(os.path.abspath(__file__))

# Проверка, присутствует ли каталог «setup» в каталоге проекта.
if "setup" in project_directory:
    # Если каталог «setup» присутствует, переместитесь на один уровень выше в родительский каталог.
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Добавление каталога проекта в начало пути поиска Python.
sys.path.insert(0, project_directory)

from library.custom_logging import setup_logging

# Настройка ведения журнала
log = setup_logging()

def check_torch():
    # Проверка nVidia toolkit or AMD toolkit
    if shutil.which('nvidia-smi') is not None or os.path.exists(
        os.path.join(
            os.environ.get('SystemRoot') or r'C:\Windows',
            'System32',
            'nvidia-smi.exe',
        )
    ):
        log.info('nVidia toolkit обнаружен')
    elif shutil.which('rocminfo') is not None or os.path.exists(
        '/opt/rocm/bin/rocminfo'
    ):
        log.info('AMD toolkit обнаружен')
    else:
        log.info('Подключение только CPU Torch')

    try:
        import torch

        log.info(f'Torch {torch.__version__}')

        # Проверка доступен ли CUDA
        if not torch.cuda.is_available():
            log.warning('Torch сообщил, что CUDA не доступен')
        else:
            if torch.version.cuda:
                # Информация версии nVidia CUDA and cuDNN
                log.info(
                    f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}'
                )
            elif torch.version.hip:
                # Информация версии AMD ROCm HIP
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning('Неизвестный Torch backend')

            # Информация о GPUs
            for device in [
                torch.cuda.device(i) for i in range(torch.cuda.device_count())
            ]:
                log.info(
                    f'Torch обнаружил GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}'
                )
                return int(torch.__version__[0])
    except Exception as e:
        log.error(f'Невозможно загрузить torch: {e}')
        sys.exit(1)


def main():
    setup_common.check_repo_version()
    # Разобрать аргументы командной строки
    parser = argparse.ArgumentParser(
        description='Validate that requirements are satisfied.'
    )
    parser.add_argument(
        '-r',
        '--requirements',
        type=str,
        help='Path to the requirements file.',
    )
    parser.add_argument('--debug', action='store_true', help='Debug on')
    args = parser.parse_args()

    torch_ver = check_torch()
    
    if args.requirements:
        setup_common.install_requirements(args.requirements, check_no_verify_flag=True)
    else:
        if torch_ver == 1:
            setup_common.install_requirements('requirements_windows_torch1.txt', check_no_verify_flag=True)
        else:
            setup_common.install_requirements('requirements_windows_torch2.txt', check_no_verify_flag=True)



if __name__ == '__main__':
    main()
