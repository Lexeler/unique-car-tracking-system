import cv2
import random
import numpy as np
import argparse
from ultralytics import YOLO

# =======================
# ===== Параметры =======
# =======================
MODEL_PATH = "yolov8m.pt"            # Путь к модели YOLO
SCALE_PERCENT = 50                   # Процент масштабирования кадра для детекции (ускоряет обработку)
IOU_LIMIT = 0.3                      # Порог IoU для сопоставления детекций с отслеживаемыми объектами
DETECTION_INTERVAL = 1               # Детекция каждые N-ый кадр
CAR_CLASS_INDEX = 2                  # Индекс автомобиля в COCO (автомобиль имеет индекс 2)

# =======================
# ==== Функции ==========
# =======================
def iou(box1, box2):
    """
    Вычисляет показатель IoU (Intersection over Union) для двух прямоугольников.
    """
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area + 1e-5)

def random_contrasting_color():
    """
    Генерирует яркий контрастный цвет.
    """
    h = random.randint(0, 179)
    color_hsv = np.uint8([[[h, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

def main():
    parser = argparse.ArgumentParser(description="Отслеживание автомобилей в видео с помощью YOLO")
    parser.add_argument("video", help="Путь к входному видеофайлу")
    args = parser.parse_args()
    VIDEO_SOURCE = args.video

    # Загружаем модель YOLO
    model = YOLO(MODEL_PATH)
    
    # Открываем видеофайл
    video = cv2.VideoCapture(VIDEO_SOURCE)
    if not video.isOpened():
        raise Exception("Не удалось открыть видео")
    
    # Получаем исходные параметры видео
    fps = video.get(cv2.CAP_PROP_FPS)
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Словари для отслеживания машин, траекторий и цветов
    tracked_cars = {}       # id -> координаты бокса (x1, y1, x2, y2)
    colors = {}             # id -> (B, G, R) яркий цвет
    unique_car_ids = set()  # множество уникальных id
    
    new_id = 1              # Счетчик для новых машин
    frame_count = 0
    trail_overlay = None    # Оверлей для накопления траекторий
    trail_points = {}       # id -> список центров (x, y)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Инициализируем оверлей для траекторий на первом кадре
        if trail_overlay is None:
            trail_overlay = np.zeros_like(frame)
        
        frame_count += 1
        
        # Детекция объектов каждые DETECTION_INTERVAL кадров
        if frame_count % DETECTION_INTERVAL == 0:
            new_w = int(width * SCALE_PERCENT / 100)
            new_h = int(height * SCALE_PERCENT / 100)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Преобразуем кадр в BGR (обычно VideoCapture возвращает BGR)
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            results = model(frame_bgr)[0]
            
            detected_boxes = []
            for det in results.boxes.data:
                # Если объект - автомобиль (индекс CAR_CLASS_INDEX)
                if int(det[5]) == CAR_CLASS_INDEX:
                    x1, y1, x2, y2 = map(int, det[:4].tolist())
                    scale_x = width / new_w
                    scale_y = height / new_h
                    box = (int(x1 * scale_x), int(y1 * scale_y),
                           int(x2 * scale_x), int(y2 * scale_y))
                    detected_boxes.append(box)
            
            # Сопоставление детекций с уже отслеживаемыми машинами
            current_cars = {}
            for box in detected_boxes:
                car_id = None
                for id_num, old_box in tracked_cars.items():
                    if iou(box, old_box) > IOU_LIMIT:
                        car_id = id_num
                        break
                if car_id is None:
                    car_id = new_id
                    new_id += 1
                current_cars[car_id] = box
                unique_car_ids.add(car_id)
                if car_id not in colors:
                    colors[car_id] = random_contrasting_color()
                if car_id not in trail_points:
                    trail_points[car_id] = []
            tracked_cars = current_cars
        
        # Отображение результата
        cv2.imshow("Видео", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
