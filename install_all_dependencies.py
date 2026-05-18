#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎙️ AI VoiceFinder - Установка всех зависимостей
Кроссплатформенный скрипт установки Python и всех пакетов
"""

import sys
import os
import subprocess
import platform
import urllib.request
import shutil
from pathlib import Path

# Цвета для консоли
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Печать заголовка"""
    print("\n" + "=" * 70)
    print(f"🐾 {text}")
    print("=" * 70 + "\n")

def print_success(text):
    """Печать успеха"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Печать ошибки"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text):
    """Печать предупреждения"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text):
    """Печать информации"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def run_command(command, description="", check=True):
    """Запуск команды с обработкой ошибок"""
    if description:
        print(f"\n📥 {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Ошибка выполнения команды: {e}")
        return False

def check_python_version():
    """Проверка версии Python"""
    print_header("ШАГ 1/6: Проверка Python")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python версия: {version_str}")
    
    if version < (3, 8):
        print_error(f"Требуется Python 3.8 или выше!")
        print_error(f"Текущая версия: {version_str}")
        print("\nПожалуйста, обновите Python:")
        print("https://www.python.org/downloads/")
        return False
    
    print_success(f"Python {version_str} установлен")
    return True

def upgrade_pip():
    """Обновление pip"""
    print_header("ШАГ 2/6: Обновление pip")
    
    if run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Обновление pip"
    ):
        print_success("pip обновлен")
        return True
    else:
        print_error("Не удалось обновить pip")
        return False

def check_gpu():
    """Проверка наличия NVIDIA GPU"""
    print_header("ШАГ 3/6: Проверка GPU")
    
    # Проверяем nvidia-smi
    if run_command("nvidia-smi", check=False):
        print_success("NVIDIA GPU обнаружен")
        
        # Получаем информацию о GPU
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"GPU: {result.stdout.strip()}")
        except:
            pass
        
        return True
    else:
        print_warning("NVIDIA GPU не обнаружен")
        print_info("Будет установлена CPU версия PyTorch")
        return False

def install_pytorch(use_gpu=True):
    """Установка PyTorch"""
    print_header("ШАГ 3/6: Установка PyTorch")
    
    if use_gpu:
        print("📥 Установка PyTorch с поддержкой CUDA 12.1...")
        print("   Это займет несколько минут...")
        command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        print("📥 Установка PyTorch (CPU версия)...")
        print("   Это займет несколько минут...")
        command = f"{sys.executable} -m pip install torch torchvision torchaudio"
    
    if run_command(command):
        print_success("PyTorch установлен")
        
        # Проверяем установку
        print("\nПроверка PyTorch...")
        try:
            import torch
            print_success(f"PyTorch версия: {torch.__version__}")
            print_success(f"CUDA доступна: {torch.cuda.is_available()}")
            return True
        except ImportError:
            print_error("PyTorch установлен некорректно")
            return False
    else:
        print_error("Не удалось установить PyTorch")
        return False

def install_requirements():
    """Установка основных зависимостей"""
    print_header("ШАГ 4/6: Установка основных зависимостей")
    
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        print("📥 Установка из requirements.txt...")
        if run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            check=False
        ):
            print_success("Основные зависимости установлены")
            return True
        else:
            print_warning("Некоторые пакеты не установились")
            print_info("Продолжаем установку...")
    else:
        print_warning("requirements.txt не найден")
        print_info("Устанавливаем пакеты вручную...")
    
    # Устанавливаем пакеты вручную
    packages = [
        ("Pillow>=10.0.0", "Pillow"),
        ("python-docx>=0.8.11", "python-docx"),
        ("soundfile>=0.12.0", "soundfile"),
        ("numpy>=1.24.0", "numpy"),
        ("pyannote.core>=5.0.0", "pyannote.core"),
        ("omegaconf>=2.3.0", "omegaconf"),
    ]
    
    for package, name in packages:
        print(f"\nУстановка {name}...")
        run_command(
            f"{sys.executable} -m pip install {package}",
            check=False
        )
    
    print_success("Основные пакеты установлены")
    return True

def install_whisperx():
    """Установка WhisperX"""
    print_header("ШАГ 5/6: Установка WhisperX")
    
    print("📥 Установка WhisperX из GitHub...")
    print("   Это может занять несколько минут...")
    
    if run_command(
        f"{sys.executable} -m pip install git+https://github.com/m-bain/whisperx.git",
        check=False
    ):
        print_success("WhisperX установлен")
        
        # Проверяем установку
        print("\nПроверка WhisperX...")
        try:
            import whisperx
            print_success("WhisperX работает корректно")
            return True
        except ImportError:
            print_error("WhisperX установлен некорректно")
            return False
    else:
        print_error("Не удалось установить WhisperX")
        print("\nВозможные причины:")
        print("  - Нет git в системе")
        print("  - Проблемы с интернетом")
        print("\nПопробуйте установить вручную:")
        print("  pip install git+https://github.com/m-bain/whisperx.git")
        return False

def install_nemo():
    """Установка NeMo"""
    print_header("ШАГ 6/6: Установка NeMo (для диаризации спикеров)")
    
    print("NeMo используется для определения спикеров в аудио.")
    print("Установка занимает много времени (~10-15 минут).")
    print()
    
    response = input("Установить NeMo? (y/n): ").lower()
    
    if response == 'y':
        print("\n📥 Установка NeMo...")
        print("   Это займет 10-15 минут, наберитесь терпения...")
        
        if run_command(
            f"{sys.executable} -m pip install nemo_toolkit[asr]",
            check=False
        ):
            print_success("NeMo установлен")
            
            # Проверяем установку
            print("\nПроверка NeMo...")
            try:
                import nemo.collections.asr
                print_success("NeMo ASR работает корректно")
                return True
            except ImportError:
                print_error("NeMo установлен некорректно")
                return False
        else:
            print_error("Не удалось установить NeMo")
            print("\nДиаризация спикеров не будет работать.")
            print("Остальные функции приложения работают нормально.")
            return False
    else:
        print_warning("NeMo не будет установлен")
        print_info("Диаризация спикеров не будет работать")
        return False

def final_check():
    """Финальная проверка установки"""
    print_header("ФИНАЛЬНАЯ ПРОВЕРКА УСТАНОВКИ")
    
    packages = [
        ("torch", "PyTorch"),
        ("whisperx", "WhisperX"),
        ("PIL", "Pillow"),
        ("docx", "python-docx"),
        ("soundfile", "SoundFile"),
    ]
    
    all_ok = True
    
    for module, name in packages:
        try:
            __import__(module)
            print_success(f"{name} установлен")
        except ImportError:
            print_error(f"{name} не установлен")
            all_ok = False
    
    # Проверяем PyTorch и CUDA
    try:
        import torch
        print_success(f"PyTorch версия: {torch.__version__}")
        print_success(f"CUDA доступна: {torch.cuda.is_available()}")
    except:
        pass
    
    return all_ok

def print_summary():
    """Печать итоговой информации"""
    print("\n" + "=" * 70)
    print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
    print("=" * 70 + "\n")
    
    print("📦 Установленные компоненты:")
    print("   • Python 3.8+")
    print("   • PyTorch (GPU или CPU)")
    print("   • WhisperX")
    print("   • Pillow, python-docx, soundfile")
    print("   • NeMo (если выбрано)")
    print()
    
    print("📁 Полезные файлы:")
    print("   • requirements.txt - список зависимостей")
    print("   • ЗАВИСИМОСТИ_ПРОЕКТА.md - подробная документация")
    print("   • Scripts/ПРОВЕРКА_GPU.bat - проверка GPU (Windows)")
    print()
    
    print("🔧 Если что-то не работает:")
    print("   1. Проверьте версию Python: python --version")
    print("   2. Проверьте pip: python -m pip --version")
    print("   3. Проверьте CUDA: nvidia-smi")
    print("   4. Переустановите проблемный пакет")
    print()
    
    print("📞 Поддержка:")
    print("   • Проверьте ЗАВИСИМОСТИ_ПРОЕКТА.md")
    print("   • Запустите Scripts/ПРОВЕРКА_GPU.bat (Windows)")
    print()

def main():
    """Главная функция"""
    print("\n" + "=" * 70)
    print("🎙️ AI VoiceFinder - Установка всех зависимостей")
    print("=" * 70)
    print("\nЭтот скрипт установит:")
    print("  1. Проверит Python")
    print("  2. PyTorch с поддержкой CUDA")
    print("  3. WhisperX")
    print("  4. NeMo для диаризации")
    print("  5. Все остальные зависимости")
    print("\nВНИМАНИЕ: Процесс может занять 30-60 минут!")
    print("          Требуется интернет-соединение.")
    print()
    
    input("Нажмите Enter для продолжения...")
    
    # Проверяем Python
    if not check_python_version():
        return 1
    
    # Обновляем pip
    if not upgrade_pip():
        print_warning("Продолжаем без обновления pip...")
    
    # Проверяем GPU
    has_gpu = check_gpu()
    
    if has_gpu:
        response = input("\nУстановить GPU версию PyTorch? (y/n): ").lower()
        use_gpu = response == 'y'
    else:
        use_gpu = False
    
    # Устанавливаем PyTorch
    if not install_pytorch(use_gpu):
        print_error("Критическая ошибка: PyTorch не установлен")
        return 1
    
    # Устанавливаем основные зависимости
    install_requirements()
    
    # Устанавливаем WhisperX
    if not install_whisperx():
        print_warning("WhisperX не установлен, но можно продолжить")
    
    # Устанавливаем NeMo
    install_nemo()
    
    # Финальная проверка
    if final_check():
        print("\n" + "=" * 70)
        print("✅ УСТАНОВКА ЗАВЕРШЕНА! 🐾")
        print("=" * 70)
        print("\nВсе зависимости установлены успешно!")
        print("\nСледующие шаги:")
        print("  1. Запустите приложение: python 'AI VoiceFinder.py'")
        print("  2. Или используйте launcher: python launcher.py")
        print("  3. Проверьте GPU: Scripts/ПРОВЕРКА_GPU.bat (Windows)")
        print()
    else:
        print("\n" + "=" * 70)
        print("⚠️  УСТАНОВКА ЗАВЕРШЕНА С ПРЕДУПРЕЖДЕНИЯМИ")
        print("=" * 70)
        print("\nНекоторые пакеты установлены некорректно.")
        print("Проверьте вывод выше и переустановите проблемные пакеты.")
        print()
    
    # Печатаем итоговую информацию
    print_summary()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Установка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
