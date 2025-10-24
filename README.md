# 🧠 YOLOv8 + BoT-SORT - Отслеживание людей на видео

Этот проект выполняет **детекцию и трекинг людей** на видео с помощью модели **YOLOv8** и трекера **BoT-SORT**.  
Результат - новое видео, где каждый человек имеет свой уникальный ID.

---

## 🚀 Возможности

- Обнаружение людей на видео в реальном времени  
- Отслеживание объектов с постоянными ID (BoT-SORT)  
- Подсчёт количества людей, которые в данный момент замечены и общее количество уникальных людей  
- Отображение FPS и номера кадра  

---

## 📁 Структура проекта
   
   ```bash
   project/
   ├── main.py                 # Точка входа (CLI)
   ├── human_detect.py         # Логика YOLO + BoT-SORT
   ├── input_video/
   │   └── input.mp4           # Видео для анализа (пример)
   ├── outputs_video/
   │   └── output.mp4          # Результат обработки
   └── requirements.txt        # Зависимости проекта
   ```
---

## ⚙️ Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Zux1U/crowd_detection.git
   cd crowd_detection
   ```
2. Cоздайте виртуальное окружение:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate #Для Windows
   source .venv/bin/activate # Для Mac/Linux
   ```
3.Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
или вручную:
   ```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install ultralytics opencv-python numpy lapx
   ```

## ▶️ Запуск
🔹 Вариант 1 - запуск без аргументов
   ```
   python main.py
   ```
🔹 Вариант 2 - указать свой путь

Вы можете явно задать входной и выходной путь:
   ```
   python main.py --source "video.mp4" --save "result.mp4"
   ```
Пример:
   ```
   python main.py --source crowd.mp4 --save tracked_crowd.mp4
   ```
🧰 Аргументы командной строки
Аргумент	Описание	По умолчанию
- --source	Путь к исходному видео
- --save	Путь для сохранения результата
- --device	Устройство для вычислений (cpu, 0, 1, …)
