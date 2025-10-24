import os
import time
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

TRK_DIR = os.path.join(
    os.path.dirname(ultralytics.__file__), "cfg", "trackers",
)
BOTSORT_YAML = os.path.join(TRK_DIR, "botsort.yaml")


def draw_corner_box(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    color_box: Tuple[int, int, int] = (0, 255, 0),
    color_corners: Tuple[int, int, int] = (255, 0, 0),
    t: int = 2, corner: int = 14,
) -> None:
    """Рисует прямоугольник с уголками вокруг объекта.

    Args:
        img (np.ndarray): Изображение (в формате BGR), на котором рисуем.
        x1 (int): Координата X верхнего левого угла.
        y1 (int): Координата Y верхнего левого угла.
        x2 (int): Координата X нижнего правого угла.
        y2 (int): Координата Y нижнего правого угла.
        color_box (Tuple[int, int, int]): Цвет рамки (BGR).
        color_corners (Tuple[int, int, int]): Цвет уголков (BGR).
        t (int): Толщина линий.
        corner (int): Длина уголков в пикселях.

    Returns:
        None
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color_box, t)
    cv2.line(img, (x1, y1), (x1 + corner, y1), color_corners, t)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color_corners, t)
    cv2.line(img, (x2, y1), (x2 - corner, y1), color_corners, t)
    cv2.line(img, (x2, y1), (x2, y1 + corner), color_corners, t)
    cv2.line(img, (x1, y2), (x1 + corner, y2), color_corners, t)
    cv2.line(img, (x1, y2), (x1, y2 - corner), color_corners, t)
    cv2.line(img, (x2, y2), (x2 - corner, y2), color_corners, t)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color_corners, t)


def draw_header(
    frame: np.ndarray, visible: int, unique: int, fps: float, frame_idx: int,
) -> None:
    """Рисует информационную панель поверх кадра.

    Args:
        frame (np.ndarray): Текущий кадр видео (в формате BGR).
        visible (int): Количество людей, видимых на кадре.
        unique (int): Количество уникальных ID за время работы.
        fps (float): Текущая оценка FPS.
        frame_idx (int): Номер текущего кадра.

    Returns:
        None
    """
    cv2.putText(
        frame, f"People visible: {visible}", (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Total unique: {unique}", (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA,
    )

    txt = f"FPS : {fps:>4.1f} | Frame : {frame_idx}"
    (tw, th), _ = cv2.getTextSize(
        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    height, width = frame.shape[:2]
    x0 = width - tw - 20

    cv2.rectangle(
        frame, (x0 - 6, 12), (x0 + tw + 6, 12 + th + 12),(255, 255, 255), -1,
    )
    cv2.putText(
        frame, txt, (x0, 12 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )


def run_singlecam_botsort_styled(
    source: str,
    save_path: str,
    device: Optional[str] = None,
) -> None:
    """Выполняет детекцию и трекинг людей на видео с помощью YOLOv8 и BoT-SORT.

    Args:
        source (str): Путь к исходному видео или URL/RTSP-потоку.
        save_path (str): Путь, по которому будет сохранён результат (MP4).
        device (Optional[str]): Устройство для вычислений
            ('cpu', '0', '1' и т.д.). По умолчанию None (автовыбор).

    Returns:
        None
    """
    # Параметры по умолчанию
    model_name = "yolov8l.pt"
    conf = 0.45
    imgsz = 896
    iou = 0.5
    tracker_yaml = BOTSORT_YAML
    color_box = (0, 255, 0)
    color_corners = (255, 0, 0)
    thickness = 2
    corner_len = 14

    # Загрузка модели
    model = YOLO(model_name)

    # Проверка входного видео
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Подготовка записи
    os.makedirs(
        os.path.dirname(os.path.abspath(save_path)),
        exist_ok=True,
    )
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps_in),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать файл: {save_path}")

    unique_ids: Set[int] = set()
    frame_idx = 0
    fps_val = 0.0
    t_last = time.time()
    fcount = 0

    # Поток трекинга
    stream = model.track(
        source=source,
        tracker=tracker_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        classes=[0],  # только человек
        stream=True,
        persist=True,
        verbose=False,
    )

    # Основной цикл
    for res in stream:
        frame_idx += 1
        frame = res.orig_img.copy()
        active_ids: Set[int] = set()

        if res.boxes is not None and getattr(res.boxes, "id", None) is not None:
            ids_np = res.boxes.id.cpu().numpy().astype(int).tolist()
            active_ids = set(int(i) for i in ids_np)
        unique_ids |= active_ids

        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            ids_arr = (
                res.boxes.id.cpu().numpy().astype(int)
                if getattr(res.boxes, "id", None) is not None else None
            )
            confs = (
                res.boxes.conf.cpu().numpy()
                if getattr(res.boxes, "conf", None) is not None else None
            )

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                draw_corner_box(
                    frame, x1, y1, x2, y2, color_box=color_box,
                    color_corners=color_corners, t=thickness, corner=corner_len,
                )
                label = (
                    f"ID: {int(ids_arr[i])}" if ids_arr is not None else "ID:-"
                )
                if confs is not None:
                    label += f"  {float(confs[i]) * 100:>.1f}%"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                y_text = max(y1 - 6, th + 6)
                cv2.rectangle(
                    frame, (x1, y_text - th - 2),
                    (x1 + tw + 8, y_text + 4), (0, 0, 0), -1,
                )
                cv2.putText(
                    frame, label, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA,
                )

        draw_header(
            frame, visible=len(active_ids), unique=len(unique_ids),
            fps=fps_val, frame_idx=frame_idx,
        )

        fcount += 1
        now = time.time()
        if now - t_last >= 0.5:
            fps_val = fcount / (now - t_last)
            fcount = 0
            t_last = now

        writer.write(frame)

    writer.release()
    print(f"Готово. Сохранено: {save_path}")
