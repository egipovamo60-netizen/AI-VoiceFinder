#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

print("=" * 70)
print("🔍 ПРОВЕРКА GPU")
print("=" * 70)
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    
    # Получаем информацию о GPU
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        gpu_memory = props.total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} ГБ)")
    
    print("\n✅ GPU успешно определена!")
else:
    print("\n⚠️ GPU не обнаружена")
    print("\nВозможные причины:")
    print("1. Драйвер NVIDIA не установлен")
    print("2. CUDA Toolkit не установлен")
    print("3. Видеокарта не NVIDIA")

print("=" * 70)
