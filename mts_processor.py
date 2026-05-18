# -*- coding: utf-8 -*-
"""
🎙️ Модуль обработки видеофайлов MTS для AI VoiceFinder
Извлекает аудиопоток из видеофайлов MTS и конвертирует их в WAV
"""

import os
import subprocess
import shutil
from pathlib import Path


class MTSProcessorError(Exception):
    """Исключение для ошибок обработки MTS файлов"""
    pass


class MTSProcessor:
    """
    Класс для обработки видеофайлов MTS (AVCHD format)
    Извлекает аудиопоток и конвертирует в WAV формат
    """
    
    SUPPORTED_FORMATS = ['.mts', '.m2ts', '.MTS', '.M2TS']
    
    def __init__(self):
        """Инициализация процессора"""
        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            raise MTSProcessorError(
                "❌ FFmpeg не найден! Установите FFmpeg для обработки видеофайлов.\n"
                "Скачать можно с: https://ffmpeg.org/download.html"
            )
    
    @staticmethod
    def _find_ffmpeg():
        """
        Ищет ffmpeg в системе
        Возвращает путь к ffmpeg или None если не найден
        """
        # Пытаемся найти ffmpeg в PATH
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            return ffmpeg
        
        # Проверяем стандартные пути для Windows
        if os.name == 'nt':
            common_paths = [
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        return None
    
    @staticmethod
    def is_mts_file(file_path):
        """
        Проверяет, является ли файл MTS файлом
        
        Args:
            file_path (str): Путь к файлу
        
        Returns:
            bool: True если файл - MTS, иначе False
        """
        _, ext = os.path.splitext(file_path)
        return ext.lower() in [ext.lower() for ext in MTSProcessor.SUPPORTED_FORMATS]
    
    def extract_audio(self, mts_file, output_wav=None, progress_callback=None):
        """
        Извлекает аудиопоток из MTS файла и сохраняет как WAV
        
        Args:
            mts_file (str): Путь к MTS файлу
            output_wav (str, optional): Путь для сохранения WAV. 
                                       Если не указан, используется имя MTS файла с расширением .wav
            progress_callback (callable, optional): Функция для отчёта о прогрессе
        
        Returns:
            str: Путь к созданному WAV файлу
        
        Raises:
            MTSProcessorError: Если не удалось извлечь аудио
        """
        
        # Проверяем существование файла
        if not os.path.exists(mts_file):
            raise MTSProcessorError(f"❌ Файл не найден: {mts_file}")
        
        # Проверяем, что это MTS файл
        if not self.is_mts_file(mts_file):
            raise MTSProcessorError(f"❌ Файл не является MTS видеофайлом: {mts_file}")
        
        # Определяем путь вывода
        if output_wav is None:
            base_name = os.path.splitext(mts_file)[0]
            output_wav = f"{base_name}.wav"
        
        # Проверяем, не существует ли уже файл
        if os.path.exists(output_wav):
            counter = 1
            base_output = output_wav
            while os.path.exists(output_wav):
                name, ext = os.path.splitext(base_output)
                output_wav = f"{name}_{counter}{ext}"
                counter += 1
        
        print(f"🎬 Обработка MTS файла: {os.path.basename(mts_file)}")
        print(f"📝 Входной файл: {mts_file}")
        print(f"📝 Размер: {os.path.getsize(mts_file) / (1024**3):.2f} ГБ")
        
        try:
            # Команда для извлечения аудио
            # Используем качественные параметры для аудио
            cmd = [
                self.ffmpeg_path,
                '-i', mts_file,
                '-vn',  # Не обрабатываем видео
                '-acodec', 'pcm_s16le',  # PCM 16-bit LE кодек
                '-ar', '48000',  # 48 kHz (стандартно для MTS)
                '-ac', '2',  # 2 канала (стерео)
                '-y',  # Перезаписать без подтверждения
                output_wav
            ]
            
            print(f"⏳ Извлечение аудиопотока...")
            if progress_callback:
                progress_callback("Извлечение аудиопотока из видео...", 0)
            
            # Запускаем ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Ждем завершения
            stdout, stderr = process.communicate()
            
            # Проверяем результат
            if process.returncode != 0:
                error_msg = stderr if stderr else stdout
                raise MTSProcessorError(
                    f"❌ Ошибка FFmpeg при извлечении аудио:\n{error_msg}"
                )
            
            # Проверяем, что файл был создан
            if not os.path.exists(output_wav):
                raise MTSProcessorError(
                    f"❌ Не удалось создать WAV файл: {output_wav}"
                )
            
            output_size = os.path.getsize(output_wav) / (1024**2)  # В МБ
            print(f"✅ Аудио извлечено успешно!")
            print(f"📄 Файл: {os.path.basename(output_wav)}")
            print(f"📊 Размер: {output_size:.2f} МБ")
            
            if progress_callback:
                progress_callback("Аудиопоток извлечен успешно!", 100)
            
            return output_wav
            
        except subprocess.CalledProcessError as e:
            raise MTSProcessorError(
                f"❌ Ошибка при вызове FFmpeg: {str(e)}"
            )
        except Exception as e:
            # Пытаемся удалить неполный файл
            try:
                if os.path.exists(output_wav):
                    os.remove(output_wav)
            except:
                pass
            raise MTSProcessorError(
                f"❌ Не удалось обработать MTS файл: {str(e)}"
            )
    
    def batch_extract(self, mts_files, output_dir=None, progress_callback=None):
        """
        Обрабатывает несколько MTS файлов
        
        Args:
            mts_files (list): Список путей к MTS файлам
            output_dir (str, optional): Директория для сохранения WAV файлов
            progress_callback (callable, optional): Функция для отчёта о прогрессе
        
        Returns:
            list: Список путей к созданным WAV файлам
        """
        
        if not mts_files:
            raise MTSProcessorError("❌ Не указаны MTS файлы для обработки")
        
        wav_files = []
        total_files = len(mts_files)
        
        print(f"\n📦 Обработка {total_files} MTS файлов...")
        
        for idx, mts_file in enumerate(mts_files, 1):
            try:
                # Определяем путь вывода
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(mts_file))[0]
                    output_wav = os.path.join(output_dir, f"{base_name}.wav")
                else:
                    output_wav = None
                
                print(f"\n📹 Файл {idx}/{total_files}")
                
                # Извлекаем аудио
                wav_file = self.extract_audio(mts_file, output_wav, progress_callback)
                wav_files.append(wav_file)
                
            except MTSProcessorError as e:
                print(f"⚠️ Ошибка при обработке файла: {e}")
                if progress_callback:
                    progress_callback(f"Ошибка: {str(e)}", -1)
                continue
        
        print(f"\n✅ Обработка завершена!")
        print(f"✓ Успешно: {len(wav_files)}/{total_files} файлов")
        
        return wav_files
    
    @staticmethod
    def get_video_info(mts_file):
        """
        Получает информацию о MTS видеофайле
        
        Args:
            mts_file (str): Путь к MTS файлу
        
        Returns:
            dict: Словарь с информацией о видеофайле
        """
        try:
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                return {"error": "FFmpeg не найден"}
            
            cmd = [ffmpeg, '-i', mts_file]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            output = stderr + stdout
            
            info = {
                "file": os.path.basename(mts_file),
                "size_gb": os.path.getsize(mts_file) / (1024**3),
                "raw_output": output[:500]  # Первые 500 символов
            }
            
            # Пытаемся найти информацию о видео и аудио
            if "Video:" in output:
                info["has_video"] = True
            if "Audio:" in output:
                info["has_audio"] = True
                # Ищем информацию о частоте дискретизации
                if "48000 Hz" in output:
                    info["sample_rate"] = 48000
                elif "44100 Hz" in output:
                    info["sample_rate"] = 44100
            
            return info
            
        except Exception as e:
            return {"error": str(e)}


def setup_ffmpeg_guide():
    """Возвращает инструкцию по установке FFmpeg"""
    return """
╔═══════════════════════════════════════════════════════════════╗
║  УСТАНОВКА FFMPEG ДЛЯ ОБРАБОТКИ MTS ВИДЕОФАЙЛОВ             ║
╚═══════════════════════════════════════════════════════════════╝

📥 СПОСОБ 1: Скачать готовый файл (рекомендуется)
  1. Перейти на https://ffmpeg.org/download.html
  2. Выбрать Windows версию
  3. Скачать "Full" версию (с кодеками)
  4. Распаковать в папку, например: C:\\ffmpeg\\
  5. Добавить путь в переменную окружения PATH

📦 СПОСОБ 2: Через пакетный менеджер (Windows)
  Если установлен chocolatey:
  > choco install ffmpeg

📦 СПОСОБ 3: Через Python пакет
  > pip install ffmpeg-python

⚙️ ПРОВЕРКА УСТАНОВКИ:
  Откройте PowerShell и выполните:
  > ffmpeg -version
  
  Если выведется версия - FFmpeg установлен корректно ✓

📌 ВАЖНО:
  - MTS файлы требуют FFmpeg с поддержкой MPEG-2 видео кодека
  - "Full" версия FFmpeg включает все необходимые кодеки
  - На распаковку и установку требуется несколько минут
"""


# For testing и debug
if __name__ == "__main__":
    print("🎬 Модуль обработки MTS видеофайлов")
    print("=" * 60)
    
    try:
        processor = MTSProcessor()
        print(f"✓ FFmpeg найден: {processor.ffmpeg_path}")
    except MTSProcessorError as e:
        print(f"❌ {e}")
        print(setup_ffmpeg_guide())
