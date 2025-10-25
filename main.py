import argparse
import os

from human_detect import run_singlecam_botsort_styled


def _project_path(*parts: str) -> str:
    """Возвращает путь внутри проекта, относительно текущего файла.

    Args:
        *parts (str): Последовательность сегментов пути.

    Returns:
        str: Абсолютный путь внутри проекта.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *parts)


# Дефолтные пути (положи своё видео сюда: assets/input.mp4)
DEFAULT_SOURCE = _project_path("input_video", "crowd.mp4")
DEFAULT_SAVE = _project_path("outputs_video", "output.mp4")


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки.

    Args:
        None

    Returns:
        argparse.Namespace: Объект с атрибутами source, save и device.
    """
    parser = argparse.ArgumentParser(
        description="Отслеживание людей на видео (YOLOv8 + BoT-SORT)."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Путь к исходному видеофайлу или RTSP/URL-потоку. "
             f"По умолчанию: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--save",
        default=DEFAULT_SAVE,
        help=f"Путь для сохранения результирующего видео (MP4). "
             f"По умолчанию: {DEFAULT_SAVE}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство для вычислений (например, '0' для GPU или 'cpu').",
    )
    return parser.parse_args()


def ensure_defaults(source_path: str, save_path: str) -> None:
    """Готовит дефолтные директории и проверяет наличие входного файла.

    Args:
        source_path (str): Путь к исходному файлу.
        save_path (str): Путь сохранения результата.

    Returns:
        None
    """
    # Создадим папку для выходного файла (если её нет)
    out_dir = os.path.dirname(os.path.abspath(save_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Если используется дефолтный путь и файла нет — подсказка пользователю
    if source_path == DEFAULT_SOURCE and not os.path.isfile(source_path):
        assets_dir = os.path.dirname(DEFAULT_SOURCE)
        os.makedirs(assets_dir, exist_ok=True)
        raise FileNotFoundError(
            f"Не найден входной файл по умолчанию:\n  {source_path}\n\n"
            f"Положи своё видео сюда и запусти снова, либо укажи путь вручную:\n"
            f"  python main.py --source /путь/к/видео.mp4 --save {DEFAULT_SAVE}"
        )


def main() -> None:
    """Основная функция запуска трекинга.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    ensure_defaults(args.source, args.save)

    run_singlecam_botsort_styled(
        source=args.source,
        save_path=args.save,
        device=args.device,
    )


if __name__ == "__main__":
    main()
