# -*- coding: utf-8 -*-
"""
🎙️ Приложение для распознавания речи AI VoiceFinder
✨ Красивый современный интерфейс
Поддержка: временные метки слов, определение спикеров
"""

print("🚀 Запуск приложения AI VoiceFinder...")
print("=" * 70)

print("📦 Импорт tkinter...")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
print("✓ tkinter импортирован")

print("🧵 Импорт threading...")
import threading
print("✓ threading импортирован")

print("🔥 Импорт torch...")
import torch
print(f"✓ torch импортирован (версия: {torch.__version__})")

print("🎙️ Импорт whisperx...")
import whisperx
print("✓ whisperx импортирован")

print("📁 Импорт os...")
import os
print("✓ os импортирован")

print("🔧 Импорт sys...")
import sys
print("✓ sys импортирован")

print("🖼️ Импорт PIL...")
from PIL import Image, ImageTk
print("✓ PIL импортирован")

print("=" * 70)
print("✅ Все модули импортированы успешно!\n")

# 📦 Настройка локального кэша для портативности
# Устанавливаем путь к кэшу в папке проекта
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# Устанавливаем переменные окружения для Hugging Face
os.environ['HF_HOME'] = LOCAL_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(LOCAL_CACHE_DIR, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(LOCAL_CACHE_DIR, 'transformers')

# Для torch hub (если используется)
os.environ['TORCH_HOME'] = os.path.join(LOCAL_CACHE_DIR, 'torch')

print(f"📦 Локальный кэш моделей: {LOCAL_CACHE_DIR}")
print("✓ Все модели будут храниться в папке проекта")


def format_timestamp(seconds):
    """Форматирование времени в формат часы:минуты:секунды"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# 🎨 Цветовая схема (гармонирует с теплыми тонами GIF)
COLORS = {
    'primary': '#D4A574',      # Золотисто-бежевый (из GIF)
    'success': '#90C695',      # Мягкий зеленый
    'warning': '#E8B86D',      # Теплый золотой
    'danger': '#D97777',       # Мягкий красный
    'dark': '#1A1410',         # Очень темный коричневый
    'darker': '#0D0A08',       # Почти черный с коричневым оттенком
    'light': '#E8DDD0',        # Светлый бежевый
    'bg': '#1A1410',           # Основной фон (темно-коричневый)
    'text': '#E8DDD0',         # Основной текст (светло-бежевый)
    'accent': '#C9A882',       # Песочный акцент
    'card_bg': '#2A2218',      # Фон карточек (темно-коричневый)
    'border': '#4A3A28',       # Границы (коричневый)
    'input_bg': '#1F1812',     # Фон полей ввода (очень темный)
    'hover': '#3A2E20',        # Цвет при наведении
}


class WhisperXApp:
    """🎙️ Главный класс приложения для распознавания речи AI VoiceFinder"""
    
    def __init__(self, root):
        print("🎨 Инициализация приложения...")
        self.root = root
        self.root.title("🎙️ AI VoiceFinder")
        self.root.geometry("320x750")
        
        # Запрещаем изменение размера окна
        self.root.resizable(False, False)
        
        # Центрируем окно на экране
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 320) // 2
        y = (screen_height - 750) // 2
        self.root.geometry(f"320x750+{x}+{y}")
        
        # Устанавливаем цвет фона
        self.root.configure(bg=COLORS['bg'])
        
        # Настраиваем стили
        self.setup_styles()
        
        print("✓ Окно создано")
        
        # Переменные
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.audio_file_path = None
        self.audio_files_list = []  # Список файлов для пакетной обработки
        self.is_processing = False
        self.device = None
        self.compute_type = None
        print("✓ Переменные инициализированы")
        
        # Создаем интерфейс
        print("🎨 Создание интерфейса...")
        self.create_widgets()
        print("✓ Интерфейс создан")
        
        # Проверяем доступность CUDA
        print("🔍 Проверка CUDA...")
        self.check_cuda()
        print("✓ Проверка CUDA завершена")
        print("\n🎉 Приложение готово к работе!")
    
    def setup_styles(self):
        """� ННастройка темной темы"""
        style = ttk.Style()
        
        # Используем современную тему
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Настраиваем стили для кнопок
        style.configure('Primary.TButton',
                       background=COLORS['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=10,
                       font=('Segoe UI', 12, 'bold'))
        
        style.map('Primary.TButton',
                 background=[('active', '#4AA5E5'), ('pressed', '#3A85C5'), ('disabled', '#3D3D3D')],
                 foreground=[('disabled', '#666666')])
        
        style.configure('Success.TButton',
                       background=COLORS['success'],
                       foreground='white',
                       borderwidth=0,
                       padding=8,
                       font=('Segoe UI', 11))
        
        style.map('Success.TButton',
                 background=[('active', '#7FDF97'), ('pressed', '#5FBF87')])
        
        # Стили для фреймов
        style.configure('Card.TFrame',
                       background=COLORS['card_bg'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('TLabelframe',
                       background=COLORS['card_bg'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=COLORS['border'])
        
        style.configure('TLabelframe.Label',
                       background=COLORS['card_bg'],
                       foreground=COLORS['light'],
                       font=('Segoe UI', 13, 'bold'))
        
        # Стили для лейблов
        style.configure('TLabel',
                       background=COLORS['card_bg'],
                       foreground=COLORS['text'])
        
        # Стили для комбобоксов
        style.configure('TCombobox',
                       fieldbackground=COLORS['input_bg'],
                       background=COLORS['card_bg'],
                       foreground=COLORS['text'],
                       borderwidth=1,
                       arrowsize=15)
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', COLORS['input_bg'])],
                 selectbackground=[('readonly', COLORS['primary'])],
                 selectforeground=[('readonly', 'white')])
        
        # Стили для чекбоксов
        style.configure('TCheckbutton',
                       background=COLORS['card_bg'],
                       foreground=COLORS['text'],
                       font=('Segoe UI', 10))
        
        style.map('TCheckbutton',
                 background=[('active', COLORS['card_bg'])])
        
        # Стили для прогресс-бара
        style.configure('TProgressbar',
                       background=COLORS['primary'],
                       troughcolor=COLORS['border'],
                       borderwidth=0,
                       thickness=20)
    
    def create_widgets(self):
        """✨ Создание красивых элементов интерфейса"""
        
        # Компактный заголовок
        title_frame = tk.Frame(self.root, bg=COLORS['darker'], height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎙️ AI VoiceFinder",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS['darker'],
            fg=COLORS['primary']
        )
        title_label.pack(expand=True)
        
        # Контейнер для карточек (компактный)
        main_container = tk.Frame(self.root, bg=COLORS['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Компактный фрейм для модели
        model_frame = ttk.LabelFrame(main_container, text="🤖 Модель", padding="8")
        model_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Статус модели и устройства (вертикально)
        status_container = tk.Frame(model_frame, bg=COLORS['card_bg'])
        status_container.pack(fill=tk.X)
        
        self.model_status_label = ttk.Label(
            status_container,
            text="Модель не загружена",
            foreground=COLORS['danger']
        )
        self.model_status_label.pack(anchor='w', pady=2)
        
        self.device_label = ttk.Label(status_container, text="", foreground=COLORS['text'])
        self.device_label.pack(anchor='w', pady=2)
        
        # Компактный фрейм для файлов
        audio_frame = ttk.LabelFrame(main_container, text="🎵 Файлы", padding="8")
        audio_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Кнопка добавления файла
        self.add_file_btn = ttk.Button(
            audio_frame,
            text="➕ Добавить файл в список",
            command=self.add_file_to_list,
            state=tk.DISABLED,
            style='Primary.TButton'
        )
        self.add_file_btn.pack(fill=tk.X, pady=2)
        
        # Список файлов с прокруткой
        list_frame = tk.Frame(audio_frame, bg=COLORS['card_bg'])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.files_listbox = tk.Listbox(
            list_frame,
            height=4,
            bg=COLORS['input_bg'],
            fg=COLORS['text'],
            selectbackground=COLORS['primary'],
            selectforeground='white',
            font=('Segoe UI', 9),
            yscrollcommand=scrollbar.set
        )
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.files_listbox.yview)
        
        # Кнопки управления списком
        list_buttons_frame = tk.Frame(audio_frame, bg=COLORS['card_bg'])
        list_buttons_frame.pack(fill=tk.X, pady=2)
        
        self.remove_file_btn = ttk.Button(
            list_buttons_frame,
            text="❌ Удалить выбранный",
            command=self.remove_selected_file,
            state=tk.DISABLED
        )
        self.remove_file_btn.pack(side=tk.LEFT, padx=(0, 2))
        
        self.clear_files_btn = ttk.Button(
            list_buttons_frame,
            text="🗑️ Очистить список",
            command=self.clear_files_list,
            state=tk.DISABLED
        )
        self.clear_files_btn.pack(side=tk.LEFT)
        
        # Счетчик файлов
        self.files_count_label = ttk.Label(
            audio_frame,
            text="Файлов: 0",
            foreground=COLORS['text']
        )
        self.files_count_label.pack(anchor='w', pady=2)
        
        # Компактный фрейм для настроек (вертикально)
        settings_frame = ttk.LabelFrame(main_container, text="⚙️ Настройки", padding="8")
        settings_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Язык
        lang_frame = tk.Frame(settings_frame, bg=COLORS['card_bg'])
        lang_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lang_frame, text="Язык:").pack(side=tk.LEFT, padx=2)
        self.language_var = tk.StringVar(value="ru")
        language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=["ru", "en", "auto"],
            state="readonly",
            width=8
        )
        language_combo.pack(side=tk.LEFT, padx=2)
        
        # Опции (вертикально)
        self.align_var = tk.BooleanVar(value=True)
        self.align_checkbox = ttk.Checkbutton(
            settings_frame,
            text="Точные временные метки",
            variable=self.align_var,
            command=self.on_align_changed
        )
        self.align_checkbox.pack(anchor='w', pady=2)
        
        self.diarize_var = tk.BooleanVar(value=False)
        self.diarize_checkbox = ttk.Checkbutton(
            settings_frame,
            text="Определение спикеров (требует временные метки)",
            variable=self.diarize_var,
            command=self.on_diarize_changed
        )
        self.diarize_checkbox.pack(anchor='w', pady=2)
        
        self.transcript_var = tk.BooleanVar(value=False)
        self.transcript_checkbox = ttk.Checkbutton(
            settings_frame,
            text="Режим стенограммы (только текст)",
            variable=self.transcript_var,
            command=self.on_transcript_changed
        )
        self.transcript_checkbox.pack(anchor='w', pady=2)
        
        # Устанавливаем правильное состояние чекбокса диаризации при инициализации
        self.on_align_changed()
        
        # Кнопка распознавания (вертикально)
        process_frame = tk.Frame(main_container, bg=COLORS['bg'])
        process_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.process_btn = ttk.Button(
            process_frame,
            text="🚀 Распознать речь",
            command=self.process_audio,
            state=tk.DISABLED,
            style='Primary.TButton'
        )
        self.process_btn.pack(fill=tk.X, pady=2)
        
        # Контейнер для GIF "Work in progress" (центрировано)
        self.progress_container = tk.Frame(process_frame, bg=COLORS['bg'])
        self.progress_container.pack(pady=5)
        
        # Загружаем GIF "Work in progress"
        try:
            gif_path = os.path.join("Sources", "Work in progress.gif")
            if os.path.exists(gif_path):
                gif_image = Image.open(gif_path)
                
                # Размер GIF 173x173 (увеличено на 50% от 115)
                original_width = gif_image.width
                original_height = gif_image.height
                scale = min(173 / original_width, 173 / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                # Загружаем все кадры GIF
                self.progress_gif_frames = []
                try:
                    while True:
                        frame_image = gif_image.copy()
                        frame_image = frame_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        self.progress_gif_frames.append(ImageTk.PhotoImage(frame_image))
                        gif_image.seek(len(self.progress_gif_frames))
                except EOFError:
                    pass
                
                # Создаем лейбл для GIF (изначально скрыт)
                self.progress_gif_label = tk.Label(
                    self.progress_container,
                    bg=COLORS['bg'],
                    bd=0
                )
                
                # Индекс текущего кадра
                self.progress_gif_index = [0]
                self.progress_gif_running = [False]
                
                print(f"✓ GIF 'Work in progress' загружен ({len(self.progress_gif_frames)} кадров)")
            else:
                print("⚠ GIF 'Work in progress' не найден")
                # Создаем обычный прогресс-бар как запасной вариант
                self.progress = ttk.Progressbar(
                    self.progress_container,
                    mode='indeterminate',
                    length=300
                )
        except Exception as e:
            print(f"⚠ Ошибка загрузки GIF: {e}")
            # Создаем обычный прогресс-бар как запасной вариант
            self.progress = ttk.Progressbar(
                self.progress_container,
                mode='indeterminate',
                length=300
            )
        
        # Статус обработки (центрировано)
        self.status_label = tk.Label(
            process_frame,
            text="Готов к работе",
            foreground=COLORS['success'],
            bg=COLORS['bg'],
            font=('Segoe UI', 11)
        )
        self.status_label.pack(pady=2)
        
        # Контейнер для кнопок результатов
        self.results_buttons_frame = tk.Frame(self.progress_container, bg=COLORS['bg'])
        
        # Кнопка открытия папки результатов (квадратная с иконкой)
        self.open_folder_btn = tk.Button(
            self.results_buttons_frame,
            text="📁",
            command=self.open_results_folder,
            bg=COLORS['success'],
            fg='white',
            font=("Segoe UI", 24),
            relief=tk.FLAT,
            cursor="hand2",
            width=3,
            height=1,
            activebackground='#7FDF97',
            activeforeground='white'
        )
        
        # Кнопка экспорта в Word (квадратная, синяя с белой буквой W)
        self.export_word_btn = tk.Button(
            self.results_buttons_frame,
            text="W",
            command=self.export_to_word,
            bg='#2B579A',  # Синий цвет Word
            fg='white',
            font=("Segoe UI", 24, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            width=3,
            height=1,
            activebackground='#1F4587',
            activeforeground='white'
        )
        # Не показываем кнопки до завершения обработки
        
        # Переменная для хранения пути к файлу результатов
        self.result_file_path = None
        # Список всех обработанных файлов (для пакетной обработки)
        self.processed_files = []  # [(audio_file, result_txt_file), ...]
        
        # Создаем пустое текстовое поле для совместимости (скрыто)
        self.result_text = scrolledtext.ScrolledText(main_container)
        # Не показываем его
    
    def open_result_file(self):
        """Открытие файла с результатами распознавания"""
        if self.result_file_path and os.path.exists(self.result_file_path):
            try:
                os.startfile(self.result_file_path)
                print(f"✓ Файл открыт: {self.result_file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
        else:
            messagebox.showwarning("Предупреждение", "Файл результатов не найден")
    
    def start_progress_animation(self):
        """Запуск анимации GIF 'Work in progress'"""
        if hasattr(self, 'progress_gif_frames') and len(self.progress_gif_frames) > 0:
            self.progress_gif_running[0] = True
            self.progress_gif_label.pack()
            self.animate_progress_gif()
        elif hasattr(self, 'progress'):
            # Запасной вариант - обычный прогресс-бар
            self.progress.pack()
            self.progress.start()
    
    def stop_progress_animation(self):
        """Остановка анимации GIF 'Work in progress'"""
        if hasattr(self, 'progress_gif_frames') and len(self.progress_gif_frames) > 0:
            self.progress_gif_running[0] = False
            self.progress_gif_label.pack_forget()
        elif hasattr(self, 'progress'):
            # Запасной вариант - обычный прогресс-бар
            self.progress.stop()
            self.progress.pack_forget()
    
    def animate_progress_gif(self):
        """Анимация GIF кадр за кадром"""
        try:
            if self.progress_gif_running[0] and hasattr(self, 'progress_gif_label'):
                if self.progress_gif_label.winfo_exists():
                    frame = self.progress_gif_frames[self.progress_gif_index[0]]
                    self.progress_gif_label.config(image=frame)
                    self.progress_gif_label.image = frame
                    self.progress_gif_index[0] = (self.progress_gif_index[0] + 1) % len(self.progress_gif_frames)
                    self.root.after(50, self.animate_progress_gif)
        except:
            pass
    
    def on_align_changed(self):
        """Обработка изменения чекбокса точных временных меток"""
        if self.align_var.get():
            # Если включены временные метки - разрешаем диаризацию
            self.diarize_checkbox.config(state=tk.NORMAL)
        else:
            # Если выключены временные метки - отключаем диаризацию и стенограмму
            self.diarize_var.set(False)
            self.diarize_checkbox.config(state=tk.DISABLED)
            self.transcript_var.set(False)
            self.transcript_checkbox.config(state=tk.DISABLED)
        
        # Обновляем состояние стенограммы
        self.on_diarize_changed()
    
    def on_diarize_changed(self):
        """Обработка изменения чекбокса диаризации"""
        if self.diarize_var.get():
            # Если включена диаризация - разрешаем стенограмму
            self.transcript_checkbox.config(state=tk.NORMAL)
        else:
            # Если выключена диаризация - отключаем стенограмму
            self.transcript_var.set(False)
            self.transcript_checkbox.config(state=tk.DISABLED)
    
    def on_transcript_changed(self):
        """Обработка изменения чекбокса режима стенограммы"""
        # Стенограмма просто меняет формат вывода, не влияет на другие настройки
        pass
    
    def check_cuda(self):
        """Проверка доступности CUDA"""
        print("\n" + "=" * 70)
        print("🔍 ПРОВЕРКА GPU")
        print("=" * 70)
        
        # Проверяем доступность CUDA
        cuda_available = torch.cuda.is_available()
        
        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступна: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"cuDNN версия: {torch.backends.cudnn.version()}")
            print(f"Количество GPU: {torch.cuda.device_count()}")
            
            # Получаем информацию о GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                gpu_memory = props.total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} ГБ)")
        
        print("=" * 70)
        
        # Используем GPU если доступен
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            self.device_label.config(
                text=f"🎮 GPU: {device_name} ({gpu_memory:.1f} ГБ)",
                foreground=COLORS['success']
            )
            self.device = "cuda"
            self.compute_type = "float16"
            print(f"\n✅ Используется GPU: {device_name}")
            print(f"Тип вычислений: {self.compute_type}")
            print(f"Ожидаемое ускорение: 10-20x по сравнению с CPU\n")
        else:
            self.device_label.config(
                text="💻 CPU режим (GPU не обнаружен)",
                foreground=COLORS['warning']
            )
            self.device = "cpu"
            self.compute_type = "int8"
            
            print("\n⚠️ GPU не обнаружен - используется CPU")
            print("\nВозможные причины:")
            print("1. Драйвер NVIDIA не установлен")
            print("2. CUDA Toolkit не установлен")
            print("3. PyTorch установлен без поддержки CUDA")
            print("\nДля диагностики запустите: ПРОВЕРКА_GPU.bat")
            print("\nРешение:")
            print("1. Установите драйвер NVIDIA")
            print("2. Установите CUDA Toolkit 12.x")
            print("3. Переустановите PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\nПриложение будет работать на CPU (медленнее в 10-20 раз)\n")
    
    def load_model(self):
        """Загрузка модели AI VoiceFinder"""
        if self.is_processing:
            messagebox.showwarning("Предупреждение", "Дождитесь завершения обработки")
            return
        
        thread = threading.Thread(target=self._load_model_thread)
        thread.start()
    
    def _load_model_thread(self):
        """Загрузка модели в отдельном потоке"""
        try:
            self.model_status_label.config(text="⏳ Загрузка модели...", foreground=COLORS['warning'])
            
            model_size = "large-v3"  # Фиксированный размер модели
            
            print(f"\nЗагрузка модели AI VoiceFinder ({model_size})...")
            print(f"Устройство: {self.device}")
            print(f"Тип вычислений: {self.compute_type}")
            
            # Загружаем модель WhisperX
            self.model = whisperx.load_model(
                model_size,
                self.device,
                compute_type=self.compute_type
            )
            
            print("✓ Модель загружена")
            
            # Выводим рекомендации по оптимизации
            if self.device == "cuda":
                print("\n" + "=" * 70)
                print("⚡ ОПТИМИЗАЦИЯ ДЛЯ GPU")
                print("=" * 70)
                print(f"Batch size для транскрипции: 16 (оптимизировано для 8 ГБ VRAM)")
                print(f"Ожидаемое ускорение: ~2-3x по сравнению с batch_size=4")
                print("=" * 70 + "\n")
            
            self.model_status_label.config(text="✅ Модель загружена", foreground=COLORS['success'])
            self.add_file_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo(
                "Успех",
                f"AI VoiceFinder ({model_size}) загружена!\nУстройство: {self.device.upper()}"
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            print("\n" + "=" * 70)
            print("ОШИБКА ЗАГРУЗКИ МОДЕЛИ")
            print("=" * 70)
            print(error_details)
            print("=" * 70)
            
            self.model_status_label.config(text="❌ Ошибка загрузки", foreground=COLORS['danger'])
            
            error_msg = str(e)
            if "Pipeline" in error_msg:
                msg = "Ошибка импорта Pipeline.\n\n" \
                      "Попробуйте:\n" \
                      "1. pip uninstall pyannote.audio\n" \
                      "2. pip install pyannote.audio\n\n" \
                      "Или используйте без диаризации."
            elif "whisperx" in error_msg.lower():
                msg = "Ошибка WhisperX.\n\n" \
                      "Переустановите:\n" \
                      "pip install git+https://github.com/m-bain/whisperx.git"
            else:
                msg = f"Не удалось загрузить модель:\n{error_msg}\n\nПодробности в консоли."
            
            messagebox.showerror("Ошибка", msg)
    
    def add_file_to_list(self):
        """Добавление файла в список для обработки"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл для добавления в список",
            filetypes=[
                ("Аудио и видео", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("Аудиофайлы", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus"),
                ("Видеофайлы", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            # Добавляем файл в список (без дубликатов)
            if file_path not in self.audio_files_list:
                self.audio_files_list.append(file_path)
                self.update_files_list()
            else:
                messagebox.showinfo("Информация", "Этот файл уже в списке")
    
    def update_files_list(self):
        """Обновление списка файлов в интерфейсе"""
        # Очищаем listbox
        self.files_listbox.delete(0, tk.END)
        
        # Добавляем файлы
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        for file_path in self.audio_files_list:
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in video_extensions:
                display_name = f"🎬 {filename}"
            else:
                display_name = f"🎵 {filename}"
            
            self.files_listbox.insert(tk.END, display_name)
        
        # Обновляем счетчик
        count = len(self.audio_files_list)
        self.files_count_label.config(text=f"Файлов: {count}")
        
        # Активируем/деактивируем кнопки
        if count > 0:
            self.process_btn.config(state=tk.NORMAL)
            self.remove_file_btn.config(state=tk.NORMAL)
            self.clear_files_btn.config(state=tk.NORMAL)
        else:
            self.process_btn.config(state=tk.DISABLED)
            self.remove_file_btn.config(state=tk.DISABLED)
            self.clear_files_btn.config(state=tk.DISABLED)
    
    def remove_selected_file(self):
        """Удаление выбранного файла из списка"""
        selection = self.files_listbox.curselection()
        if selection:
            index = selection[0]
            del self.audio_files_list[index]
            self.update_files_list()
    
    def clear_files_list(self):
        """Очистка списка файлов"""
        self.audio_files_list = []
        self.update_files_list()
    
    def extract_audio_from_video(self, video_path):
        """Извлечение аудио из видеофайла"""
        try:
            import subprocess
            import tempfile
            
            print(f"\n🎬 Извлечение аудио из видео: {os.path.basename(video_path)}")
            
            # Создаем временный файл для аудио
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Используем ffmpeg для извлечения аудио
            # -i: входной файл
            # -vn: без видео
            # -acodec pcm_s16le: аудио кодек WAV
            # -ar 16000: частота дискретизации 16kHz (для Whisper)
            # -ac 1: моно
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # Без видео
                '-acodec', 'pcm_s16le',  # WAV формат
                '-ar', '16000',  # 16kHz
                '-ac', '1',  # Моно
                '-y',  # Перезаписать если существует
                temp_audio_path
            ]
            
            print("Запуск ffmpeg...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 минут максимум
            )
            
            if result.returncode != 0:
                raise Exception(f"Ошибка ffmpeg: {result.stderr}")
            
            print(f"✓ Аудио извлечено: {temp_audio_path}")
            return temp_audio_path
            
        except FileNotFoundError:
            raise Exception(
                "ffmpeg не найден!\n\n"
                "Для работы с видео нужен ffmpeg.\n\n"
                "Установка:\n"
                "1. Скачайте ffmpeg: https://ffmpeg.org/download.html\n"
                "2. Добавьте в PATH\n"
                "3. Или положите ffmpeg.exe в папку приложения"
            )
        except subprocess.TimeoutExpired:
            raise Exception("Превышено время ожидания извлечения аудио (5 минут)")
        except Exception as e:
            raise Exception(f"Ошибка извлечения аудио: {str(e)}")
    
    def process_audio(self):
        """Запуск распознавания речи"""
        if not self.model:
            messagebox.showwarning("Предупреждение", "Сначала загрузите модель")
            return
        
        # Проверяем есть ли файлы для обработки
        if not self.audio_files_list:
            messagebox.showwarning("Предупреждение", "Сначала добавьте файл(ы) в список")
            return
        
        if self.is_processing:
            messagebox.showwarning("Предупреждение", "Обработка уже выполняется")
            return
        
        # Очищаем список обработанных файлов перед новой обработкой
        self.processed_files = []
        print("✓ Список обработанных файлов очищен")
        
        # Пакетная обработка
        files_count = len(self.audio_files_list)
        
        if files_count == 1:
            # Один файл - обрабатываем сразу
            thread = threading.Thread(target=self._process_batch_thread)
            thread.start()
        else:
            # Несколько файлов - спрашиваем подтверждение
            response = messagebox.askyesno(
                "Пакетная обработка",
                f"Обработать {files_count} файлов?\n\n"
                f"Это может занять продолжительное время."
            )
            if not response:
                return
            
            thread = threading.Thread(target=self._process_batch_thread)
            thread.start()
    
    def _process_audio_thread(self):
        """Обработка аудио в отдельном потоке"""
        temp_audio_path = None  # Инициализируем переменную для временного файла
        
        try:
            # Очищаем память GPU перед началом
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Память GPU очищена")
            
            self.is_processing = True
            self.process_btn.config(state=tk.DISABLED)
            self.add_file_btn.config(state=tk.DISABLED)
            
            # Скрываем кнопки результатов при начале новой обработки
            self.results_buttons_frame.pack_forget()
            self.open_folder_btn.pack_forget()
            self.export_word_btn.pack_forget()
            
            # Показываем статус обратно (если был скрыт)
            self.status_label.pack(pady=(10, 0))
            
            # Запускаем анимацию GIF вместо прогресс-бара
            self.start_progress_animation()
            
            # Создаем файл для сохранения
            save_filepath = self._create_save_file()
            
            # Сохраняем путь к файлу результатов
            self.result_file_path = save_filepath
            
            # НЕ открываем файл сразу - это замедляет обработку
            # Файл откроется автоматически после завершения
            
            # Очищаем текстовое поле
            self.result_text.delete(1.0, tk.END)
            
            # Получаем параметры
            language = self.language_var.get()
            if language == "auto":
                language = None
            
            # Проверяем является ли файл видео
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(self.audio_file_path)[1].lower()
            
            if file_ext in video_extensions:
                # Извлекаем аудио из видео
                self.status_label.config(text="🎬 Извлечение аудио из видео...", foreground=COLORS['warning'])
                self.root.update()
                
                try:
                    temp_audio_path = self.extract_audio_from_video(self.audio_file_path)
                    audio_path = temp_audio_path
                except Exception as e:
                    self.status_label.config(text="❌ Ошибка", foreground=COLORS['danger'])
                    messagebox.showerror("Ошибка", str(e))
                    return
            else:
                audio_path = self.audio_file_path
            
            # ШАГ 1: ПОТОКОВОЕ Распознавание речи
            self.status_label.config(text="🐾 Распознавание речи... 0%", foreground=COLORS['warning'])
            self.root.update()
            
            print("\n" + "=" * 70)
            print("ШАГ 1: ПОТОКОВОЕ Распознавание речи")
            print("=" * 70)
            
            # Загружаем аудио
            audio = whisperx.load_audio(audio_path)
            
            # Заголовок в интерфейсе и файле
            header = "РАСПОЗНАННЫЙ ТЕКСТ (потоковый режим):\n" + "=" * 70 + "\n\n"
            self.result_text.insert(tk.END, header)
            self._append_to_file(save_filepath, header)
            self.root.update()
            
            # ПОТОКОВАЯ ТРАНСКРИПЦИЯ - обрабатываем по частям
            print("Начало потоковой транскрипции...")
            
            # Определяем оптимальные параметры в зависимости от устройства
            if self.device == "cuda":
                # Для GPU 8 ГБ: максимальная производительность
                batch_size = 32  # Увеличено до 32 для максимальной скорости
                chunk_length = 60  # Увеличено до 60 секунд для меньшего overhead
                print(f"🎮 GPU режим: batch_size={batch_size}, chunk_length={chunk_length}s")
                
                # Включаем оптимизации для GPU
                torch.backends.cudnn.benchmark = True  # Автоматический выбор лучших алгоритмов
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True  # Ускорение матричных операций
                print("✓ GPU оптимизации включены")
            else:
                # Для CPU: оптимальные значения
                batch_size = 8  # Увеличено с 4 до 8
                chunk_length = 45  # Увеличено до 45 секунд
                print(f"💻 CPU режим: batch_size={batch_size}, chunk_length={chunk_length}s")
            
            # Разбиваем аудио на части для потоковой обработки
            sample_rate = 16000
            chunk_samples = chunk_length * sample_rate
            total_samples = len(audio)
            num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            
            print(f"Длина аудио: {total_samples / sample_rate:.1f} секунд")
            print(f"Разбито на {num_chunks} частей по {chunk_length} секунд")
            
            all_segments = []
            buffer_lines = []  # Буфер для накопления строк перед записью
            
            for chunk_idx in range(num_chunks):
                start_sample = chunk_idx * chunk_samples
                end_sample = min((chunk_idx + 1) * chunk_samples, total_samples)
                audio_chunk = audio[start_sample:end_sample]
                
                # Обрабатываем часть с оптимальным batch_size
                # Для GPU также можно использовать num_workers для ускорения загрузки данных
                if self.device == "cuda":
                    chunk_result = self.model.transcribe(
                        audio_chunk, 
                        batch_size=batch_size, 
                        language=language,
                        num_workers=2  # Параллельная загрузка данных
                    )
                else:
                    chunk_result = self.model.transcribe(audio_chunk, batch_size=batch_size, language=language)
                
                if "segments" in chunk_result:
                    # Корректируем временные метки с учетом смещения
                    time_offset = start_sample / sample_rate
                    for seg in chunk_result["segments"]:
                        seg["start"] += time_offset
                        seg["end"] += time_offset
                        all_segments.append(seg)
                        
                        # СРАЗУ ВЫВОДИМ
                        text = seg.get("text", "")
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        
                        line = f"[{format_timestamp(start)} - {format_timestamp(end)}]: {text}\n"
                        
                        # Выводим в интерфейс
                        self.result_text.insert(tk.END, line)
                        self.result_text.see(tk.END)
                        
                        # Добавляем в буфер вместо немедленной записи
                        buffer_lines.append(line)
                
                # Записываем буфер в файл после каждого чанка (не после каждого сегмента!)
                if buffer_lines:
                    self._append_to_file(save_filepath, "".join(buffer_lines))
                    buffer_lines.clear()
                
                # Обновляем статус только после каждого чанка
                progress = ((chunk_idx + 1) / num_chunks) * 100
                self.status_label.config(
                    text=f"🐾 Распознавание речи... {progress:.0f}%",
                    foreground=COLORS['warning']
                )
                self.root.update()
                print(f"✓ Обработана часть {chunk_idx + 1}/{num_chunks}")
            
            # Собираем результат
            result = {"segments": all_segments, "language": language if language else "ru"}
            
            print(f"✓ Распознано {len(all_segments)} сегментов")
            self._append_to_file(save_filepath, "\n")
            
            # ШАГ 2: Выравнивание (точные временные метки)
            if self.align_var.get():
                self.status_label.config(text="⚡ Выравнивание... 0%", foreground=COLORS['primary'])
                self.root.update()
                
                print("\n" + "=" * 70)
                print("ШАГ 2: Выравнивание (точные временные метки)")
                print("=" * 70)
                
                # Загружаем модель выравнивания если еще не загружена
                if self.align_model is None:
                    # Определяем язык из результата
                    detected_lang = result.get("language", language if language else "ru")
                    print(f"Определенный язык: {detected_lang}")
                    print(f"Загрузка модели выравнивания для языка: {detected_lang}")
                    
                    try:
                        self.align_model, self.align_metadata = whisperx.load_align_model(
                            language_code=detected_lang,
                            device=self.device
                        )
                        print("✓ Модель выравнивания загружена")
                    except Exception as e:
                        print(f"⚠ Ошибка загрузки модели выравнивания: {e}")
                        print("Продолжаем без выравнивания...")
                        self.align_var.set(False)
                
                if self.align_model:
                    self.status_label.config(text="⚡ Выравнивание... 50%", foreground=COLORS['primary'])
                    self.root.update()
                    
                    aligned_result = whisperx.align(
                        result["segments"],
                        self.align_model,
                        self.align_metadata,
                        audio,
                        self.device,
                        return_char_alignments=False
                    )
                    
                    self.status_label.config(text="⚡ Выравнивание... 100%", foreground=COLORS['primary'])
                    self.root.update()
                    
                    # Обновляем сегменты, сохраняя остальные данные
                    result["segments"] = aligned_result["segments"]
                    if "word_segments" in aligned_result:
                        result["word_segments"] = aligned_result["word_segments"]
                    print(f"✓ Выравнивание завершено")
                    print(f"✓ Сегментов после выравнивания: {len(result['segments'])}")
            
            # ШАГ 3: Диаризация (определение спикеров)
            print(f"\n🔍 Проверка диаризации: diarize_var={self.diarize_var.get()}, align_var={self.align_var.get()}")
            
            if self.diarize_var.get() and self.align_var.get():
                self.status_label.config(text="👥 Определение спикеров... 0%", foreground=COLORS['accent'])
                self.root.update()
                
                print("\n" + "=" * 70)
                print("ШАГ 3: Диаризация (определение спикеров)")
                print("=" * 70)
                print(f"Диаризация ВКЛЮЧЕНА - начинаем обработку")
                
                try:
                    if self.diarize_model is None:
                        self.status_label.config(text="👥 Загрузка модели диаризации... 10%", foreground=COLORS['accent'])
                        self.root.update()
                        
                        print("Загрузка модели диаризации NeMo...")
                        print("✓ NeMo не требует токен Hugging Face!")
                        
                        # Импортируем NeMo
                        import nemo.collections.asr as nemo_asr
                        from omegaconf import OmegaConf
                        
                        # Создаем конфигурацию для диаризации
                        # Создаем полную конфигурацию для NeMo
                        config = {
                            'manifest_filepath': None,
                            'out_dir': os.path.join(os.path.dirname(__file__), 'nemo_outputs'),
                            'oracle_vad': False,
                            'device': self.device,
                            'num_workers': 0,  # Важно для Windows
                            'sample_rate': 16000,  # Частота дискретизации
                            'batch_size': 1,
                            'verbose': True,  # Вывод прогресса
                            'diarizer': {
                                'manifest_filepath': None,
                                'out_dir': os.path.join(os.path.dirname(__file__), 'nemo_outputs'),
                                'oracle_vad': False,
                                'device': self.device,
                                'collar': 0.25,  # Допуск для границ сегментов (в секундах)
                                'ignore_overlap': True,  # Игнорировать перекрывающиеся сегменты
                                'clustering': {
                                    'parameters': {
                                        'oracle_num_speakers': False,
                                        'max_num_speakers': 8,
                                        'enhanced_count_thres': 80,
                                        'max_rp_threshold': 0.25,
                                        'sparse_search_volume': 30
                                    }
                                },
                                'vad': {
                                    'model_path': 'vad_multilingual_marblenet',
                                    'parameters': {
                                        'window_length_in_sec': 0.15,
                                        'shift_length_in_sec': 0.01,
                                        'smoothing': 'median',
                                        'overlap': 0.5,
                                        'onset': 0.8,
                                        'offset': 0.6,
                                        'pad_onset': 0.05,
                                        'pad_offset': 0.05,
                                        'min_duration_on': 0.2,
                                        'min_duration_off': 0.2
                                    }
                                },
                                'speaker_embeddings': {
                                    'model_path': 'titanet_large',
                                    'parameters': {
                                        'window_length_in_sec': 1.5,
                                        'shift_length_in_sec': 0.75,
                                        'multiscale_weights': [1, 1, 1, 1, 1],
                                        'save_embeddings': False
                                    }
                                }
                            }
                        }
                        
                        cfg = OmegaConf.create(config)
                        
                        # Загружаем модель диаризации NeMo
                        self.diarize_model = nemo_asr.models.ClusteringDiarizer(cfg=cfg)
                        print("✓ Модель диаризации NeMo загружена")
                    
                    self.status_label.config(text="👥 Подготовка аудио... 20%", foreground=COLORS['accent'])
                    self.root.update()
                    
                    # ОБХОДНОЙ ПУТЬ: Конвертируем аудио во временный файл
                    print("Подготовка аудио для диаризации...")
                    import tempfile
                    import soundfile as sf
                    import json
                    
                    # Создаем временную директорию для NeMo
                    temp_dir = tempfile.mkdtemp()
                    
                    # Создаем временный WAV файл с правильными параметрами
                    temp_audio_path = os.path.join(temp_dir, "audio.wav")
                    sf.write(temp_audio_path, audio, 16000)
                    print(f"✓ Временный файл создан: {temp_audio_path}")
                    
                    # Создаем manifest файл для NeMo
                    manifest_path = os.path.join(temp_dir, "manifest.json")
                    audio_duration = len(audio) / 16000
                    manifest_entry = {
                        "audio_filepath": temp_audio_path,
                        "offset": 0,
                        "duration": audio_duration,
                        "label": "infer",
                        "text": "-",
                        "num_speakers": None,
                        "rttm_filepath": None,
                        "uem_filepath": None
                    }
                    
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest_entry, f)
                        f.write('\n')
                    
                    print(f"✓ Manifest создан: {manifest_path}")
                    
                    try:
                        self.status_label.config(text="👥 Анализ аудио... 40%", foreground=COLORS['accent'])
                        self.root.update()
                        
                        # Обновляем конфигурацию модели с путями к файлам
                        self.diarize_model._cfg.diarizer.manifest_filepath = manifest_path
                        self.diarize_model._cfg.diarizer.out_dir = temp_dir
                        
                        # Применяем диаризацию
                        print("Запуск диаризации...")
                        self.status_label.config(text="👥 Определение спикеров... 60%", foreground=COLORS['accent'])
                        self.root.update()
                        
                        self.diarize_model.diarize()
                        
                        self.status_label.config(text="👥 Обработка результатов... 80%", foreground=COLORS['accent'])
                        self.root.update()
                        
                        print("✓ Диаризация выполнена")
                        
                        # Читаем результаты из RTTM файла
                        from pyannote.core import Annotation, Segment
                        rttm_path = os.path.join(temp_dir, "pred_rttms", "audio.rttm")
                        
                        if not os.path.exists(rttm_path):
                            raise FileNotFoundError(f"RTTM файл не найден: {rttm_path}")
                        
                        # Парсим RTTM файл
                        diarize_segments = Annotation()
                        speakers = set()
                        
                        with open(rttm_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 8:
                                    start_time = float(parts[3])
                                    duration = float(parts[4])
                                    speaker = parts[7]
                                    speakers.add(speaker)
                                    diarize_segments[Segment(start_time, start_time + duration)] = speaker
                        
                        print(f"✓ Найдено спикеров: {len(speakers)}")
                        
                        # Проверяем структуру result
                        segments_list = None
                        if isinstance(result, dict):
                            segments_list = result.get("segments", [])
                        elif isinstance(result, list):
                            segments_list = result
                        
                        # Проверяем что у сегментов есть слова (после выравнивания)
                        if self.align_var.get() and segments_list:
                            # Присваиваем спикеров к словам
                            print("Присвоение спикеров к словам...")
                            
                            # Создаем правильную структуру для assign_word_speakers
                            transcript_with_words = {"segments": segments_list}
                            
                            try:
                                diarized_result = whisperx.assign_word_speakers(diarize_segments, transcript_with_words)
                                print(f"✓ Диаризация завершена")
                                
                                # Проверяем что спикеры действительно присвоены
                                speakers_found = set()
                                if "segments" in diarized_result:
                                    for seg in diarized_result["segments"]:
                                        if "speaker" in seg:
                                            speakers_found.add(seg["speaker"])
                                        if "words" in seg:
                                            for word in seg["words"]:
                                                if "speaker" in word:
                                                    speakers_found.add(word["speaker"])
                                
                                if speakers_found:
                                    print(f"✓ Найдено спикеров в результате: {len(speakers_found)} - {speakers_found}")
                                    # Обновляем сегменты в основном результате
                                    result["segments"] = diarized_result["segments"]
                                    
                                    self.status_label.config(text="👥 Определение спикеров... 100%", foreground=COLORS['accent'])
                                    self.root.update()
                                else:
                                    print("⚠ Спикеры не найдены в результате, используем альтернативный метод")
                                    raise Exception("Спикеры не присвоены")
                            except Exception as e:
                                print(f"⚠ Ошибка при присвоении спикеров через WhisperX: {e}")
                                print("Использую альтернативный метод...")
                                
                                # Альтернативный метод - присваиваем спикеров вручную
                                for segment in segments_list:
                                    if "words" in segment:
                                        for word in segment["words"]:
                                            word_start = word.get("start", 0)
                                            word_end = word.get("end", 0)
                                            word_mid = (word_start + word_end) / 2
                                            
                                            # Находим спикера в это время
                                            for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                                if turn.start <= word_mid <= turn.end:
                                                    word["speaker"] = speaker
                                                    break
                                    
                                    # Также присваиваем спикера всему сегменту
                                    seg_start = segment.get("start", 0)
                                    seg_end = segment.get("end", 0)
                                    seg_mid = (seg_start + seg_end) / 2
                                    
                                    for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                        if turn.start <= seg_mid <= turn.end:
                                            segment["speaker"] = speaker
                                            break
                                
                                result = {"segments": segments_list}
                                print(f"✓ Спикеры присвоены альтернативным методом")
                                
                                self.status_label.config(text="👥 Определение спикеров... 100%", foreground=COLORS['accent'])
                                self.root.update()
                        else:
                            # Если нет выравнивания, просто добавляем информацию о спикерах к сегментам
                            print("⚠ Для лучшей диаризации включите 'Точные временные метки'")
                            print("Добавление спикеров к сегментам...")
                            
                            # Простое присвоение спикеров по времени
                            for segment in segments_list:
                                seg_start = segment.get("start", 0)
                                seg_end = segment.get("end", 0)
                                seg_mid = (seg_start + seg_end) / 2
                                
                                # Находим спикера в это время
                                for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                    if turn.start <= seg_mid <= turn.end:
                                        segment["speaker"] = speaker
                                        break
                            
                            result = {"segments": segments_list}
                            print(f"✓ Спикеры добавлены к сегментам")
                            
                            self.status_label.config(text="👥 Определение спикеров... 100%", foreground=COLORS['accent'])
                            self.root.update()
                        
                    finally:
                        # Удаляем временные файлы
                        import shutil
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                            print("✓ Временные файлы удалены")
                    
                except Exception as e:
                    import traceback
                    print(f"\n⚠ Ошибка диаризации: {e}")
                    print("Детали ошибки:")
                    traceback.print_exc()
                    
                    # Показываем более информативное сообщение
                    error_msg = str(e)
                    if "Sizes of tensors" in error_msg:
                        msg = "Ошибка обработки аудио.\n\nВозможные причины:\n" \
                              "- Нестандартная длина аудио\n" \
                              "- Поврежденный файл\n" \
                              "- Несовместимый формат\n\n" \
                              "Попробуйте:\n" \
                              "- Конвертировать в WAV\n" \
                              "- Использовать другой файл\n" \
                              "- Отключить диаризацию"
                    else:
                        msg = f"Диаризация недоступна:\n{error_msg}\n\n" \
                              "Продолжаем без определения спикеров."
                    
                    messagebox.showwarning("Предупреждение", msg)
            elif self.diarize_var.get() and not self.align_var.get():
                print("\n⚠️ Диаризация требует включения точных временных меток")
                print("Диаризация пропущена")
            else:
                print(f"\n⚠️ Диаризация ОТКЛЮЧЕНА (diarize={self.diarize_var.get()}, align={self.align_var.get()})")
            
            # Вывод результатов
            self.status_label.config(text="💾 Сохранение результатов... 0%", foreground=COLORS['primary'])
            self.root.update()
            
            self._display_results(result, save_filepath)
            
            # Добавляем время завершения в файл
            self._finalize_save_file(save_filepath)
            
            print("\n" + "=" * 70)
            print(f"✓ Результат сохранен: {save_filepath}")
            print("=" * 70)
            
            # Добавляем в список обработанных файлов
            self.processed_files.append((self.audio_file_path, save_filepath))
            print(f"✓ Файл добавлен в список обработанных: {os.path.basename(self.audio_file_path)}")
            
            # Скрываем статус и GIF
            self.status_label.pack_forget()
            self.stop_progress_animation()
            
            # Показываем кнопки результатов в progress_container (на месте GIF, по центру, ниже на 1/3)
            self.results_buttons_frame.pack(pady=(60, 10), expand=True)
            
            # Показываем кнопку открытия папки
            self.open_folder_btn.pack(side=tk.LEFT, padx=5)
            
            # Если режим стенограммы - показываем кнопку экспорта в Word
            if self.transcript_var.get():
                self.export_word_btn.pack(side=tk.LEFT, padx=5)
            
            messagebox.showinfo("Успех", "Распознавание завершено!\n\nРезультат автоматически сохранен.")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("\n" + "=" * 70)
            print("ОШИБКА ПРИ ОБРАБОТКЕ АУДИО")
            print("=" * 70)
            print(error_details)
            print("=" * 70)
            
            # Скрываем GIF и показываем статус ошибки
            self.stop_progress_animation()
            self.status_label.pack(pady=(10, 0))
            self.status_label.config(text="❌ Ошибка", foreground=COLORS['danger'])
            messagebox.showerror("Ошибка", f"Ошибка при обработке:\n{str(e)}\n\nПодробности в консоли.")
        
        finally:
            # Удаляем временный аудиофайл если он был создан
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    print(f"✓ Временный аудиофайл удален: {temp_audio_path}")
                except Exception as e:
                    print(f"⚠ Не удалось удалить временный файл: {e}")
            
            self.is_processing = False
            
            self.process_btn.config(state=tk.NORMAL)
            self.add_file_btn.config(state=tk.NORMAL)
            self.add_file_btn.config(state=tk.NORMAL)
    
    def _process_batch_thread(self):
        """Пакетная обработка файлов"""
        total_files = len(self.audio_files_list)
        processed = 0
        failed = 0
        failed_files = []
        
        try:
            self.is_processing = True
            self.process_btn.config(state=tk.DISABLED)
            self.add_file_btn.config(state=tk.DISABLED)
            self.remove_file_btn.config(state=tk.DISABLED)
            self.clear_files_btn.config(state=tk.DISABLED)
            
            # Запускаем анимацию GIF
            self.start_progress_animation()
            
            print("\n" + "=" * 70)
            print(f"🎬 ПАКЕТНАЯ ОБРАБОТКА: {total_files} файлов")
            print("=" * 70)
            
            for index, file_path in enumerate(self.audio_files_list, 1):
                filename = os.path.basename(file_path)
                
                print(f"\n{'=' * 70}")
                print(f"📁 Файл {index}/{total_files}: {filename}")
                print(f"{'=' * 70}")
                
                # Обновляем статус
                self.status_label.config(
                    text=f"📁 Обработка {index}/{total_files}: {filename[:30]}...",
                    foreground=COLORS['warning']
                )
                self.root.update()
                
                # Временно устанавливаем текущий файл
                original_file_path = self.audio_file_path
                self.audio_file_path = file_path
                
                try:
                    # Обрабатываем файл (используем существующую логику)
                    self._process_single_file()
                    processed += 1
                    print(f"✓ Файл {index}/{total_files} обработан успешно")
                    
                except Exception as e:
                    failed += 1
                    failed_files.append(filename)
                    print(f"❌ Ошибка обработки файла {index}/{total_files}: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    # Восстанавливаем оригинальный путь
                    self.audio_file_path = original_file_path
            
            # Итоговый отчет
            print("\n" + "=" * 70)
            print("📊 ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ")
            print("=" * 70)
            print(f"Всего файлов: {total_files}")
            print(f"✓ Обработано успешно: {processed}")
            print(f"❌ Ошибок: {failed}")
            
            if failed_files:
                print("\nФайлы с ошибками:")
                for f in failed_files:
                    print(f"  - {f}")
            
            print("=" * 70)
            
            # Скрываем статус и GIF
            self.status_label.pack_forget()
            self.stop_progress_animation()
            
            # Показываем кнопки результатов в progress_container (на месте GIF, по центру, ниже на 1/3)
            self.results_buttons_frame.pack(pady=(60, 10), expand=True)
            
            # Показываем кнопку открытия папки
            self.open_folder_btn.pack(side=tk.LEFT, padx=5)
            
            # Если режим стенограммы - показываем кнопку экспорта в Word
            if self.transcript_var.get():
                self.export_word_btn.pack(side=tk.LEFT, padx=5)
            
            if failed > 0:
                messagebox.showwarning(
                    "Пакетная обработка завершена",
                    f"Обработано: {processed}/{total_files}\n"
                    f"Ошибок: {failed}\n\n"
                    f"Файлы с ошибками:\n" + "\n".join(failed_files[:5]) +
                    (f"\n... и еще {len(failed_files) - 5}" if len(failed_files) > 5 else "")
                )
            else:
                messagebox.showinfo(
                    "Пакетная обработка завершена",
                    f"Все файлы обработаны успешно!\n\n"
                    f"Обработано: {processed}/{total_files}"
                )
        
        except Exception as e:
            print(f"\n❌ Критическая ошибка пакетной обработки: {e}")
            import traceback
            traceback.print_exc()
            
            # Скрываем GIF и показываем статус ошибки
            self.stop_progress_animation()
            self.status_label.pack(pady=(10, 0))
            self.status_label.config(text="❌ Ошибка", foreground=COLORS['danger'])
            messagebox.showerror("Ошибка", f"Критическая ошибка:\n{str(e)}")
        
        finally:
            self.is_processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.add_file_btn.config(state=tk.NORMAL)
            self.add_file_btn.config(state=tk.NORMAL)
            self.remove_file_btn.config(state=tk.NORMAL)
            self.clear_files_btn.config(state=tk.NORMAL)
    
    def _process_single_file(self):
        """Обработка одного файла (используется в пакетной обработке)"""
        temp_audio_path = None
        
        try:
            # Создаем файл для сохранения
            save_filepath = self._create_save_file()
            self.result_file_path = save_filepath
            
            # Получаем параметры
            language = self.language_var.get()
            if language == "auto":
                language = None
            
            # Проверяем является ли файл видео
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(self.audio_file_path)[1].lower()
            
            if file_ext in video_extensions:
                # Извлекаем аудио из видео
                temp_audio_path = self.extract_audio_from_video(self.audio_file_path)
                audio_path = temp_audio_path
            else:
                audio_path = self.audio_file_path
            
            # Загружаем аудио
            audio = whisperx.load_audio(audio_path)
            
            # ШАГ 1: Распознавание
            print("\nШАГ 1: Распознавание речи")
            self.status_label.config(text="🐾 Распознавание речи... 0%", foreground=COLORS['warning'])
            self.root.update()
            
            # Определяем оптимальные параметры в зависимости от устройства
            if self.device == "cuda":
                batch_size = 16  # Для GPU 8 ГБ
                chunk_length = 30
            else:
                batch_size = 4
                chunk_length = 30
            
            sample_rate = 16000
            chunk_samples = chunk_length * sample_rate
            total_samples = len(audio)
            num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            
            all_segments = []
            
            for chunk_idx in range(num_chunks):
                progress = ((chunk_idx + 1) / num_chunks) * 100
                self.status_label.config(text=f"🐾 Распознавание речи... {progress:.0f}%", foreground=COLORS['warning'])
                self.root.update()
                print(f"Распознавание: {progress:.0f}%")
                
                start_sample = chunk_idx * chunk_samples
                end_sample = min((chunk_idx + 1) * chunk_samples, total_samples)
                audio_chunk = audio[start_sample:end_sample]
                
                chunk_result = self.model.transcribe(audio_chunk, batch_size=batch_size, language=language)
                
                if "segments" in chunk_result:
                    time_offset = start_sample / sample_rate
                    for seg in chunk_result["segments"]:
                        seg["start"] += time_offset
                        seg["end"] += time_offset
                        all_segments.append(seg)
            
            result = {"segments": all_segments, "language": language if language else "ru"}
            print(f"✓ Распознано {len(all_segments)} сегментов")
            
            # ШАГ 2: Выравнивание
            if self.align_var.get():
                print("\nШАГ 2: Выравнивание")
                self.status_label.config(text="⚡ Выравнивание... 0%", foreground=COLORS['primary'])
                self.root.update()
                
                if self.align_model is None:
                    detected_lang = result.get("language", language if language else "ru")
                    self.align_model, self.align_metadata = whisperx.load_align_model(
                        language_code=detected_lang,
                        device=self.device
                    )
                
                if self.align_model:
                    self.status_label.config(text="⚡ Выравнивание... 50%", foreground=COLORS['primary'])
                    self.root.update()
                    print("Выравнивание: 50%")
                    
                    aligned_result = whisperx.align(
                        result["segments"],
                        self.align_model,
                        self.align_metadata,
                        audio,
                        self.device,
                        return_char_alignments=False
                    )
                    result["segments"] = aligned_result["segments"]
                    
                    self.status_label.config(text="⚡ Выравнивание... 100%", foreground=COLORS['primary'])
                    self.root.update()
                    print(f"✓ Выравнивание завершено (100%)")
            
            # ШАГ 3: Диаризация
            print(f"\n🔍 Проверка диаризации: diarize_var={self.diarize_var.get()}, align_var={self.align_var.get()}")
            
            if self.diarize_var.get() and self.align_var.get():
                print("\nШАГ 3: Диаризация")
                self.status_label.config(text="👥 Определение спикеров... 0%", foreground=COLORS['accent'])
                self.root.update()
                
                # Здесь должна быть вся логика диаризации
                # Копируем из _process_audio_thread
                import nemo.collections.asr as nemo_asr
                from omegaconf import OmegaConf
                import tempfile
                import soundfile as sf
                import json
                from pyannote.core import Annotation, Segment
                
                if self.diarize_model is None:
                    self.status_label.config(text="👥 Загрузка модели диаризации... 10%", foreground=COLORS['accent'])
                    self.root.update()
                    print("Загрузка модели диаризации NeMo... (10%)")
                    config = {
                        'manifest_filepath': None,
                        'out_dir': os.path.join(os.path.dirname(__file__), 'nemo_outputs'),
                        'oracle_vad': False,
                        'device': self.device,
                        'num_workers': 0,
                        'sample_rate': 16000,
                        'batch_size': 1,
                        'verbose': True,
                        'diarizer': {
                            'manifest_filepath': None,
                            'out_dir': os.path.join(os.path.dirname(__file__), 'nemo_outputs'),
                            'oracle_vad': False,
                            'device': self.device,
                            'collar': 0.25,
                            'ignore_overlap': True,
                            'clustering': {
                                'parameters': {
                                    'oracle_num_speakers': False,
                                    'max_num_speakers': 8,
                                    'enhanced_count_thres': 80,
                                    'max_rp_threshold': 0.25,
                                    'sparse_search_volume': 30
                                }
                            },
                            'vad': {
                                'model_path': 'vad_multilingual_marblenet',
                                'parameters': {
                                    'window_length_in_sec': 0.15,
                                    'shift_length_in_sec': 0.01,
                                    'smoothing': 'median',
                                    'overlap': 0.5,
                                    'onset': 0.8,
                                    'offset': 0.6,
                                    'pad_onset': 0.05,
                                    'pad_offset': 0.05,
                                    'min_duration_on': 0.2,
                                    'min_duration_off': 0.2
                                }
                            },
                            'speaker_embeddings': {
                                'model_path': 'titanet_large',
                                'parameters': {
                                    'window_length_in_sec': 1.5,
                                    'shift_length_in_sec': 0.75,
                                    'multiscale_weights': [1, 1, 1, 1, 1],
                                    'save_embeddings': False
                                }
                            }
                        }
                    }
                    cfg = OmegaConf.create(config)
                    self.diarize_model = nemo_asr.models.ClusteringDiarizer(cfg=cfg)
                    print("✓ Модель диаризации загружена")
                
                # Создаем временные файлы
                self.status_label.config(text="👥 Подготовка аудио... 20%", foreground=COLORS['accent'])
                self.root.update()
                print("Подготовка аудио... (20%)")
                
                temp_dir = tempfile.mkdtemp()
                temp_audio_path_diarize = os.path.join(temp_dir, "audio.wav")
                sf.write(temp_audio_path_diarize, audio, 16000)
                
                manifest_path = os.path.join(temp_dir, "manifest.json")
                audio_duration = len(audio) / 16000
                manifest_entry = {
                    "audio_filepath": temp_audio_path_diarize,
                    "offset": 0,
                    "duration": audio_duration,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": None,
                    "uem_filepath": None
                }
                
                with open(manifest_path, 'w') as f:
                    json.dump(manifest_entry, f)
                    f.write('\n')
                
                try:
                    self.status_label.config(text="👥 Анализ аудио... 40%", foreground=COLORS['accent'])
                    self.root.update()
                    print("Анализ аудио... (40%)")
                    
                    self.diarize_model._cfg.diarizer.manifest_filepath = manifest_path
                    self.diarize_model._cfg.diarizer.out_dir = temp_dir
                    
                    self.status_label.config(text="👥 Определение спикеров... 60%", foreground=COLORS['accent'])
                    self.root.update()
                    print("Определение спикеров... (60%)")
                    
                    self.diarize_model.diarize()
                    
                    self.status_label.config(text="👥 Обработка результатов... 80%", foreground=COLORS['accent'])
                    self.root.update()
                    print("Обработка результатов... (80%)")
                    
                    rttm_path = os.path.join(temp_dir, "pred_rttms", "audio.rttm")
                    diarize_segments = Annotation()
                    speakers = set()
                    
                    with open(rttm_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                start_time = float(parts[3])
                                duration = float(parts[4])
                                speaker = parts[7]
                                speakers.add(speaker)
                                diarize_segments[Segment(start_time, start_time + duration)] = speaker
                    
                    print(f"✓ Найдено спикеров: {len(speakers)}")
                    
                    # Присваиваем спикеров
                    segments_list = result.get("segments", [])
                    transcript_with_words = {"segments": segments_list}
                    
                    try:
                        diarized_result = whisperx.assign_word_speakers(diarize_segments, transcript_with_words)
                        result["segments"] = diarized_result["segments"]
                        
                        self.status_label.config(text="👥 Определение спикеров... 100%", foreground=COLORS['accent'])
                        self.root.update()
                        print(f"✓ Спикеры присвоены (100%)")
                    except Exception as e:
                        print(f"⚠ Ошибка assign_word_speakers: {e}")
                        # Альтернативный метод
                        for segment in segments_list:
                            if "words" in segment:
                                for word in segment["words"]:
                                    word_mid = (word.get("start", 0) + word.get("end", 0)) / 2
                                    for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                        if turn.start <= word_mid <= turn.end:
                                            word["speaker"] = speaker
                                            break
                            seg_mid = (segment.get("start", 0) + segment.get("end", 0)) / 2
                            for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
                                if turn.start <= seg_mid <= turn.end:
                                    segment["speaker"] = speaker
                                    break
                        result["segments"] = segments_list
                        
                        self.status_label.config(text="👥 Определение спикеров... 100%", foreground=COLORS['accent'])
                        self.root.update()
                        print(f"✓ Спикеры присвоены альтернативным методом (100%)")
                
                finally:
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            
            # Сохраняем результаты
            self.status_label.config(text="💾 Сохранение результатов... 0%", foreground=COLORS['primary'])
            self.root.update()
            print("\nСохранение результатов...")
            
            self._display_results(result, save_filepath)
            
            # Добавляем время завершения в файл
            self._finalize_save_file(save_filepath)
            
            print(f"✓ Результат сохранен: {save_filepath} (100%)")
            
            # Добавляем в список обработанных файлов
            self.processed_files.append((self.audio_file_path, save_filepath))
            print(f"✓ Файл добавлен в список обработанных: {os.path.basename(self.audio_file_path)}")
            
        finally:
            # Удаляем временный аудиофайл если был создан
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"✓ Временный файл удален: {temp_audio_path}")
                except Exception as e:
                    print(f"⚠ Не удалось удалить временный файл: {e}")
        temp_audio_path = None
        
        try:
            # Вся логика обработки уже есть в _process_audio_thread
            # Копируем только необходимую часть
            
            # Очищаем память GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Создаем файл для сохранения
            save_filepath = self._create_save_file()
            
            # Получаем параметры
            language = self.language_var.get()
            if language == "auto":
                language = None
            
            # Проверяем видео
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            file_ext = os.path.splitext(self.audio_file_path)[1].lower()
            
            if file_ext in video_extensions:
                temp_audio_path = self.extract_audio_from_video(self.audio_file_path)
                audio_path = temp_audio_path
            else:
                audio_path = self.audio_file_path
            
            # Загружаем и обрабатываем аудио
            audio = whisperx.load_audio(audio_path)
            
            # Транскрипция
            result = self.model.transcribe(audio, batch_size=4, language=language)
            
            # Выравнивание
            if self.align_var.get() and self.align_model:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
            
            # Сохраняем результаты
            self._save_results_to_file(result, save_filepath)
            
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
    
    def _save_results_to_file(self, result, filepath):
        """Сохранение результатов в файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ\n")
            f.write("=" * 70 + "\n\n")
            
            if isinstance(result, dict) and "segments" in result:
                for segment in result["segments"]:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "")
                    f.write(f"[{format_timestamp(start)} - {format_timestamp(end)}]: {text}\n")
    
    def _display_results(self, result, save_filepath):
        """Отображение и сохранение результатов"""
        
        # Проверяем режим стенограммы
        if self.transcript_var.get():
            # Режим стенограммы - только текст без временных меток
            self._display_transcript_mode(result, save_filepath)
            return
        
        # Проверяем наличие спикеров в результате
        speakers_in_result = set()
        if "segments" in result:
            for seg in result["segments"]:
                if "speaker" in seg:
                    speakers_in_result.add(seg["speaker"])
                if "words" in seg:
                    for word in seg["words"]:
                        if "speaker" in word:
                            speakers_in_result.add(word["speaker"])
        
        if speakers_in_result:
            print(f"\n✓ Спикеры в результате для вывода: {speakers_in_result}")
        else:
            print("\n⚠ Спикеры не найдены в результате для вывода")
        
        # Заголовок для сегментов
        header = "\n" + "=" * 70 + "\nВРЕМЕННЫЕ МЕТКИ:\n" + "=" * 70 + "\n\n"
        self.result_text.insert(tk.END, header)
        self._append_to_file(save_filepath, header)
        
        if "segments" in result:
            total_segments = len(result["segments"])
            
            for i, segment in enumerate(result["segments"], 1):
                # Информация о сегменте
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "")
                speaker = segment.get("speaker", None)
                
                # Формируем строку
                if speaker:
                    line = f"\n[{format_timestamp(start)} - {format_timestamp(end)}] {speaker}:\n{text}\n"
                else:
                    line = f"\n[{format_timestamp(start)} - {format_timestamp(end)}]:\n{text}\n"
                
                # Если есть слова с временными метками
                if "words" in segment and segment["words"]:
                    line += "  Слова:\n"
                    for word_info in segment["words"]:
                        word = word_info.get("word", "")
                        word_start = word_info.get("start", 0)
                        word_end = word_info.get("end", 0)
                        word_speaker = word_info.get("speaker", "")
                        
                        if word_speaker:
                            line += f"    [{format_timestamp(word_start)} - {format_timestamp(word_end)}] {word_speaker}: {word}\n"
                        else:
                            line += f"    [{format_timestamp(word_start)} - {format_timestamp(word_end)}]: {word}\n"
                
                # Выводим в интерфейс
                self.result_text.insert(tk.END, line)
                self.result_text.see(tk.END)
                
                # Сохраняем в файл
                self._append_to_file(save_filepath, line)
                
                # Обновляем статус
                progress = (i / total_segments) * 100
                self.status_label.config(
                    text=f"💾 Сохранение результатов... {progress:.0f}%",
                    foreground=COLORS['primary']
                )
                self.root.update()
    
    def _create_save_file(self):
        """Создание файла для сохранения с заголовком"""
        try:
            from datetime import datetime
            
            # Сохраняем время начала обработки
            self.processing_start_time = datetime.now()
            
            # Создаем папку для результатов
            results_dir = os.path.join(os.path.dirname(__file__), "Results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Генерируем имя файла с временной меткой
            timestamp = self.processing_start_time.strftime("%Y-%m-%d_%H-%M-%S")
            
            # Получаем имя исходного аудиофайла
            audio_filename = os.path.basename(self.audio_file_path)
            audio_name = os.path.splitext(audio_filename)[0]
            
            # Формируем имя файла
            filename = f"{audio_name}_{timestamp}_whisperx.txt"
            filepath = os.path.join(results_dir, filename)
            
            # Создаем файл с заголовком
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"AI VoiceFinder - Распознавание речи\n")
                f.write("=" * 70 + "\n")
                f.write(f"Исходный файл: {audio_filename}\n")
                f.write(f"Время начала: {self.processing_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Модель: large-v3\n")
                f.write(f"Язык: {self.language_var.get()}\n")
                f.write(f"Режим: {'Стенограмма' if self.transcript_var.get() else 'Обычный'}\n")
                f.write(f"Точные метки: {'Да' if self.align_var.get() else 'Нет'}\n")
                f.write(f"Диаризация: {'Да' if self.diarize_var.get() else 'Нет'}\n")
                f.write("=" * 70 + "\n\n")
            
            print(f"\n✓ Создан файл: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"\n⚠ Ошибка создания файла: {e}")
            return None
    
    def _display_transcript_mode(self, result, save_filepath):
        """Отображение результатов в режиме стенограммы (текст по спикерам с временными интервалами)"""
        
        # Заголовок для стенограммы
        header = "\n" + "=" * 70 + "\nСТЕНОГРАММА:\n" + "=" * 70 + "\n\n"
        self.result_text.insert(tk.END, header)
        self._append_to_file(save_filepath, header)
        
        if "segments" in result:
            total_segments = len(result["segments"])
            
            # Группируем текст по спикерам
            current_speaker = None
            current_text = ""
            start_time = None
            end_time = None
            
            for i, segment in enumerate(result["segments"], 1):
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", None)
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", 0)
                
                # Определяем имя спикера
                if speaker:
                    speaker_name = speaker
                else:
                    speaker_name = "[спикер не определен]"
                
                if speaker_name != current_speaker:
                    # Если сменился спикер, выводим накопленный текст
                    if current_text and start_time is not None:
                        time_interval = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                        line = f"{time_interval} {current_speaker}: {current_text}\n\n"
                        self.result_text.insert(tk.END, line)
                        self._append_to_file(save_filepath, line)
                    
                    # Начинаем новый блок
                    current_speaker = speaker_name
                    current_text = text
                    start_time = seg_start
                    end_time = seg_end
                else:
                    # Тот же спикер - добавляем текст и обновляем конечное время
                    current_text += " " + text
                    end_time = seg_end
                
                # Обновляем статус
                progress = (i / total_segments) * 100
                self.status_label.config(
                    text=f"💾 Сохранение результатов... {progress:.0f}%",
                    foreground=COLORS['primary']
                )
                self.root.update()
            
            # Выводим последний блок
            if current_text and start_time is not None:
                time_interval = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                line = f"{time_interval} {current_speaker}: {current_text}\n\n"
                self.result_text.insert(tk.END, line)
                self._append_to_file(save_filepath, line)
    
    def _append_to_file(self, filepath, text):
        """Добавление текста в файл"""
        if filepath:
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
            except Exception as e:
                print(f"\n⚠ Ошибка записи в файл: {e}")
    
    def _finalize_save_file(self, filepath):
        """Добавление времени завершения в конец файла"""
        if filepath:
            try:
                from datetime import datetime
                
                end_time = datetime.now()
                
                # Вычисляем длительность обработки
                if hasattr(self, 'processing_start_time'):
                    duration = end_time - self.processing_start_time
                    duration_seconds = duration.total_seconds()
                    hours = int(duration_seconds // 3600)
                    minutes = int((duration_seconds % 3600) // 60)
                    seconds = int(duration_seconds % 60)
                    
                    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = "неизвестно"
                
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write(f"Время завершения: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Длительность обработки: {duration_str}\n")
                    f.write("=" * 70 + "\n")
                
                print(f"✓ Время завершения добавлено в файл")
            except Exception as e:
                print(f"\n⚠ Ошибка добавления времени завершения: {e}")
    
    def save_result(self):
        """Ручное сохранение результата"""
        text = self.result_text.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Предупреждение", "Нет текста для сохранения")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".txt",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                messagebox.showinfo("Успех", f"Результат сохранен в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def clear_result(self):
        """Очистка результата"""
        self.result_text.delete(1.0, tk.END)
    
    def copy_result(self):
        """Копирование результата в буфер обмена"""
        text = self.result_text.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Предупреждение", "Нет текста для копирования")
            return
        
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Успех", "Текст скопирован в буфер обмена")
    
    def open_results_folder(self):
        """Открытие папки с результатами"""
        try:
            results_dir = os.path.join(os.path.dirname(__file__), "Results")
            os.makedirs(results_dir, exist_ok=True)
            os.startfile(results_dir)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{str(e)}")
    
    def export_to_word(self):
        """Экспорт стенограммы в MS Word - создание отдельных файлов для каждого обработанного аудио"""
        # Проверяем, есть ли обработанные файлы
        if not self.processed_files:
            messagebox.showerror("Ошибка", "Нет обработанных файлов для экспорта")
            return
        
        try:
            # Импортируем библиотеку для работы с Word
            try:
                from docx import Document
                from docx.shared import Pt
            except ImportError:
                messagebox.showerror(
                    "Ошибка",
                    "Библиотека python-docx не установлена!\n\n"
                    "Установите её командой:\n"
                    "pip install python-docx"
                )
                return
            
            print(f"\n📄 Экспорт в Word ({len(self.processed_files)} файлов)...")
            
            created_files = []
            
            # Обрабатываем каждый файл из списка обработанных
            for audio_file, txt_file in self.processed_files:
                if not os.path.exists(txt_file):
                    print(f"⚠ Файл не найден: {txt_file}")
                    continue
                
                print(f"\n📝 Обработка: {os.path.basename(txt_file)}")
                
                # Читаем файл
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"✓ Прочитано {len(content)} символов")
                
                # Создаем документ Word
                doc = Document()
                
                # Разбиваем на параграфы по двойным переносам строк
                paragraphs = content.split('\n\n')
                
                print(f"✓ Найдено параграфов: {len(paragraphs)}")
                
                for para_text in paragraphs:
                    if para_text.strip():
                        # Добавляем параграф
                        p = doc.add_paragraph(para_text.strip())
                        
                        # Если это заголовок или временная метка - делаем жирным
                        if para_text.strip().startswith('['):
                            p.runs[0].bold = True
                            p.runs[0].font.size = Pt(11)
                        else:
                            p.runs[0].font.size = Pt(12)
                
                # Сохраняем документ
                word_filename = txt_file.replace('.txt', '.docx')
                doc.save(word_filename)
                
                print(f"✓ Документ Word сохранен: {word_filename}")
                created_files.append(word_filename)
            
            # Показываем результат
            if created_files:
                # Открываем все созданные Word файлы
                for word_file in created_files:
                    try:
                        os.startfile(word_file)
                        print(f"✓ Открыт: {os.path.basename(word_file)}")
                    except Exception as e:
                        print(f"⚠ Не удалось открыть {os.path.basename(word_file)}: {e}")
                
                files_list = "\n".join([f"• {os.path.basename(f)}" for f in created_files])
                messagebox.showinfo(
                    "✅ Экспорт завершен", 
                    f"Создано и открыто документов Word: {len(created_files)}\n\n{files_list}"
                )
                print(f"\n✅ Экспорт завершен! Создано и открыто {len(created_files)} файлов Word")
            else:
                messagebox.showwarning("Предупреждение", "Не удалось создать ни одного документа")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Не удалось экспортировать в Word:\n{str(e)}")


def show_splash_screen():
    """⏳ Показать заставку загрузки с GIF анимацией"""
    splash = tk.Tk()
    splash.title("Загрузка...")
    splash.overrideredirect(True)
    
    # Загружаем GIF сначала
    gif_path = os.path.join("Sources", "Load waiting.gif")
    gif_width = 400
    gif_height = 300
    
    try:
        if os.path.exists(gif_path):
            temp_gif = Image.open(gif_path)
            gif_width = int(temp_gif.width * 0.5)
            gif_height = int(temp_gif.height * 0.5)
            temp_gif.close()
    except:
        pass
    
    # Размеры окна
    width = 700
    height = max(400, gif_height + 100)
    
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    splash.geometry(f"{width}x{height}+{x}+{y}")
    splash.configure(bg=COLORS['darker'])
    
    # Главный контейнер
    main_container = tk.Frame(splash, bg=COLORS['darker'])
    main_container.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
    
    # ЛЕВАЯ ЧАСТЬ
    left_frame = tk.Frame(main_container, bg=COLORS['darker'], width=350)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 20))
    left_frame.pack_propagate(False)
    
    # Заголовок
    title_label = tk.Label(
        left_frame,
        text="🎤 AI VoiceFinder",
        font=("Segoe UI", 24, "bold"),
        bg=COLORS['darker'],
        fg=COLORS['primary'],
        anchor='w'
    )
    title_label.pack(anchor='w', pady=(20, 5))
    
    subtitle_label = tk.Label(
        left_frame,
        text="Распознавание речи с AI",
        font=("Segoe UI", 14),
        bg=COLORS['darker'],
        fg=COLORS['light'],
        anchor='w'
    )
    subtitle_label.pack(anchor='w', pady=(0, 30))
    
    # Переменная для отслеживания состояния (ВАЖНО: до animate_text!)
    splash._is_alive = True
    splash._after_ids = []
    
    # Динамичный статус с посимвольной анимацией и прокруткой
    status_frame = tk.Frame(left_frame, bg=COLORS['darker'])
    status_frame.pack(anchor='w', pady=20, fill=tk.BOTH, expand=True)
    
    status_label = tk.Label(
        status_frame,
        text="",
        font=("Consolas", 12, "bold"),
        bg=COLORS['darker'],
        fg=COLORS['text'],
        anchor='nw',
        justify='left',
        wraplength=320
    )
    status_label.pack(anchor='nw', fill=tk.BOTH, expand=True)
    
    # Счетчик процентов
    percent_label = tk.Label(
        left_frame,
        text="0%",
        font=("Segoe UI", 28, "bold"),
        bg=COLORS['darker'],
        fg=COLORS['accent'],
        anchor='w'
    )
    percent_label.pack(side=tk.BOTTOM, anchor='w', pady=(10, 0))
    
    # Текст для анимации
    loading_messages = [
        "🎙️ Загрузка модели AI VoiceFinder...",
        "📥 Модель: large-v3",
        "⚙️ Инициализация компонентов...",
        "🎵 Подготовка аудио движка...",
        "✨ Почти готово!",
        "🔧 Настройка интерфейса...",
        "💾 Проверка зависимостей...",
        "🎤 Тестирование микрофона...",
        "🚀 Запуск приложения..."
    ]
    
    completed_messages = []
    current_message = [0]
    current_char = [0]
    cursor_visible = [True]
    current_percent = [0]
    
    def animate_text():
        if not splash._is_alive:
            return
        try:
            if splash.winfo_exists():
                msg_index = current_message[0]
                char_index = current_char[0]
                
                if msg_index < len(loading_messages):
                    message = loading_messages[msg_index]
                    
                    if char_index <= len(message):
                        # Показываем текст посимвольно с курсором
                        current_text = message[:char_index]
                        cursor = "█" if cursor_visible[0] else ""
                        
                        # Показываем последние 5 строк для эффекта прокрутки с промежутками
                        display_lines = completed_messages[-4:] if len(completed_messages) > 4 else completed_messages
                        display_text = "\n".join(display_lines)  # Одинарный перенос
                        if display_text:
                            display_text += "\n"
                        display_text += current_text + cursor
                        
                        status_label.config(text=display_text)
                        
                        # Обновляем проценты
                        progress = int((msg_index * 100 + (char_index / len(message) * 100)) / len(loading_messages))
                        if progress != current_percent[0]:
                            current_percent[0] = progress
                            percent_label.config(text=f"{progress}%")
                        
                        current_char[0] += 1
                        delay = 1  # Максимально быстрая печать
                    else:
                        # Сообщение завершено - добавляем в историю
                        completed_messages.append(message)
                        current_message[0] += 1
                        current_char[0] = 0
                        delay = 150  # Короткая пауза между сообщениями
                    
                    after_id = splash.after(delay, animate_text)
                    splash._after_ids.append(after_id)
                else:
                    # Зацикливаем анимацию
                    completed_messages.clear()
                    current_message[0] = 0
                    current_char[0] = 0
                    current_percent[0] = 0
                    percent_label.config(text="0%")
                    after_id = splash.after(800, animate_text)
                    splash._after_ids.append(after_id)
        except:
            pass
    
    def blink_cursor():
        if not splash._is_alive:
            return
        try:
            if splash.winfo_exists():
                cursor_visible[0] = not cursor_visible[0]
                after_id = splash.after(400, blink_cursor)
                splash._after_ids.append(after_id)
        except:
            pass
    
    animate_text()
    blink_cursor()
    
    # ПРАВАЯ ЧАСТЬ - GIF
    right_frame = tk.Frame(main_container, bg=COLORS['darker'])
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Загружаем GIF
    try:
        if os.path.exists(gif_path):
            gif_image = Image.open(gif_path)
            original_width = gif_image.width
            original_height = gif_image.height
            new_width = int(original_width * 0.5)
            new_height = int(original_height * 0.5)
            
            gif_label = tk.Label(right_frame, bg=COLORS['darker'], bd=0)
            gif_label.pack(expand=True)
            
            frames = []
            try:
                while True:
                    frame_image = gif_image.copy()
                    if frame_image.mode != 'RGBA':
                        frame_image = frame_image.convert('RGBA')
                    frame_image = frame_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    frames.append(ImageTk.PhotoImage(frame_image))
                    gif_image.seek(len(frames))
            except EOFError:
                pass
            
            frame_index = [0]
            
            def update_frame():
                if not splash._is_alive:
                    return
                try:
                    if splash.winfo_exists():
                        frame = frames[frame_index[0]]
                        gif_label.config(image=frame)
                        gif_label.image = frame
                        frame_index[0] = (frame_index[0] + 1) % len(frames)
                        after_id = splash.after(50, update_frame)
                        splash._after_ids.append(after_id)
                except:
                    pass
            
            update_frame()
        else:
            no_gif_label = tk.Label(right_frame, text="⏳", font=("Segoe UI", 64),
                                   bg=COLORS['darker'], fg=COLORS['primary'])
            no_gif_label.pack(expand=True)
    except Exception as e:
        print(f"⚠ Не удалось загрузить GIF: {e}")
        emoji_label = tk.Label(right_frame, text="⏳", font=("Segoe UI", 64),
                              bg=COLORS['darker'], fg=COLORS['primary'])
        emoji_label.pack(expand=True)
    
    # Функция для безопасного закрытия
    def safe_close():
        splash._is_alive = False
        for after_id in splash._after_ids:
            try:
                splash.after_cancel(after_id)
            except:
                pass
        try:
            splash.destroy()
        except:
            pass
    
    splash.safe_close = safe_close
    
    splash.update()
    return splash


def main():
    """Главная функция запуска приложения"""
    try:
        # Показываем заставку загрузки
        print("\n⏳ Показываем заставку загрузки...")
        splash = show_splash_screen()
        splash.update_idletasks()
        splash.update()
        
        # Переменные для хранения загруженной модели
        loaded_model = {'model': None, 'device': None, 'compute_type': None}
        
        # Функция загрузки модели в фоновом потоке
        def load_model_background():
            try:
                print("\n🔥 Начинаем загрузку модели large-v3...")
                
                # Определяем устройство
                use_gpu = True
                if torch.cuda.is_available() and use_gpu:
                    device = "cuda"
                    compute_type = "float16"
                    device_name = torch.cuda.get_device_name(0)
                    print(f"✓ Используется GPU: {device_name}")
                else:
                    device = "cpu"
                    compute_type = "int8"
                    print("✓ Используется CPU")
                
                # Загружаем модель WhisperX large-v3
                print("⏳ Загрузка модели AI VoiceFinder (large-v3)...")
                model = whisperx.load_model(
                    "large-v3",
                    device,
                    compute_type=compute_type
                )
                print("✅ Модель large-v3 загружена успешно!")
                
                # Сохраняем модель
                loaded_model['model'] = model
                loaded_model['device'] = device
                loaded_model['compute_type'] = compute_type
                
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                import traceback
                traceback.print_exc()
        
        # Запускаем загрузку модели в отдельном потоке
        model_thread = threading.Thread(target=load_model_background, daemon=True)
        model_thread.start()
        
        # Ждем загрузки модели с обновлением заставки
        print("\n⏳ Ожидание завершения загрузки модели...")
        while model_thread.is_alive():
            splash.update()
            model_thread.join(timeout=0.1)
        
        # Закрываем заставку безопасно
        splash.safe_close()
        
        # Проверяем успешность загрузки модели
        if loaded_model['model'] is None:
            print("\n" + "=" * 70)
            print("❌ ОШИБКА: Модель не загружена!")
            print("=" * 70)
            print("Приложение не может запуститься без модели.")
            print("Проверьте ошибки выше и попробуйте снова.")
            print("=" * 70)
            input("\nНажмите Enter для выхода...")
            return
        
        print("\n✅ Модель успешно загружена, создание главного окна...")
        root = tk.Tk()
        print("✓ Главное окно создано")
        
        print("Создание экземпляра приложения...")
        app = WhisperXApp(root)
        
        # Устанавливаем загруженную модель в приложение
        app.model = loaded_model['model']
        app.device = loaded_model['device']
        app.compute_type = loaded_model['compute_type']
        app.model_status_label.config(text="✅ Модель large-v3 загружена", foreground=COLORS['success'])
        app.add_file_btn.config(state=tk.NORMAL)
        print("✓ Модель large-v3 установлена в приложение")
        
        print("✓ Экземпляр приложения создан")
        
        print("\nЗапуск главного цикла событий...")
        print("Окно должно открыться. Для выхода закройте окно.")
        print("=" * 70)
        root.mainloop()
        
        print("\n✓ Приложение закрыто нормально")
        sys.exit(0)
    except Exception as e:
        import traceback
        print("\n" + "=" * 70)
        print("КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ПРИЛОЖЕНИЯ")
        print("=" * 70)
        print(f"Ошибка: {e}")
        print("\nПолная информация об ошибке:")
        print("-" * 70)
        traceback.print_exc()
        print("=" * 70)
        input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    main()

