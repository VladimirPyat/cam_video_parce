import os
import time

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02}:{int(m):02}:{int(s):02}'


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_time(elapsed_time)
        print(f'Время выполнения: {formatted_time}')
        return result

    return wrapper


import cv2
import numpy
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# настройка параметров
CURRENT_DIR = os.path.dirname(__file__)
SOURCE_DIR = '_source'
RESULT_DIR = '_result'

CAM_AREAS = {
    '248': (0.1, 0.45, 0.55, 0.25)
}  # ключ - номер камеры. значения - кортеж координат области детекции


CLASSES_TO_DETECT = [2, 3, 7]  # ищем машины, мотоциклы, грузовики

FRAMES_SKIP = 6  # пропуск части кадров в видео для ускорения работы


def is_path_exist():
    result_path = os.path.join(CURRENT_DIR, RESULT_DIR)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    source_path = os.path.join(CURRENT_DIR, SOURCE_DIR)
    if not os.path.exists(source_path):
        return False

    return True


def image_crop(image_in, cam_id=tuple(CAM_AREAS.keys())[0]):
    # обрезаем изображение в зависимости от заданной зоны детекции
    if isinstance(image_in, numpy.ndarray):
        image = image_in
    else:
        image = cv2.imread(image_in)
    height, width, _ = image.shape
    coord_from_cam = CAM_AREAS[cam_id]
    x_percent, y_percent, w_percent, h_percent = coord_from_cam
    x, y, w, h = int(width * x_percent), int(height * y_percent), int(width * w_percent), int(height * h_percent)
    selected_area = image[y:y + h, x:x + w]
    return selected_area


def show_image(image_in):
    cv2.imshow('image to detect', image_in)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def yolo_obj_detect(image_in):
    predictions = model.predict(image_in, classes=CLASSES_TO_DETECT, save_txt=True)

    return predictions


def is_obj_detected(result):
    return True if len(result.boxes) > 0 else False


def is_obj_motion(result):
    return True  # заглушка. если машина на видео не двигается - не сохраняем кадр


@timer
def video_proceed(video_filename, cam_id=tuple(CAM_AREAS.keys())[0], show_img=False):
    if not is_path_exist():
        print(f'Путь {SOURCE_DIR} к обрабатываемым файлам не найден')
        return

    video_path = os.path.join(CURRENT_DIR, SOURCE_DIR, video_filename)
    if not os.path.exists(video_path):
        print(f'Файл {video_filename} не найден в {SOURCE_DIR}')
        return

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    detection_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        # Обработка кадра каждые FRAMES_SKIP кадров
        if frame_counter % FRAMES_SKIP == 0:
            frame_crop = image_crop(frame, cam_id)
            result = yolo_obj_detect(frame_crop)[0]

            if is_obj_detected(result) and is_obj_motion(result):
                detection_counter += 1
                frame_filename = f"{video_filename}_{frame_counter}.jpg"
                frame_path = os.path.join(CURRENT_DIR, RESULT_DIR, frame_filename)
                if show_img:
                    cv2.imshow('Frame', frame_crop)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cv2.imwrite(frame_path, frame)  # Сохранение кадра если обнаружен объект

    cap.release()
    cv2.destroyAllWindows()

    print(f'Сохранено {detection_counter} файлов в {RESULT_DIR}')


def video_path_proceed(cam_id=tuple(CAM_AREAS.keys())[0], show_img=False, files_extention='mp4'):
    video_path = os.path.join(CURRENT_DIR, SOURCE_DIR)
    if not os.path.exists(video_path):
        print(f'Путь {video_path} к обрабатываемым файлам не найден')
        return

    files_list = [file for file in os.listdir(video_path) if file.endswith(files_extention)]

    for file in files_list:
        video_proceed(file, cam_id=cam_id, show_img=show_img)







if __name__ == '__main__':

    # video_file_name = 'test.mp4'
    # video_proceed(video_file_name)

    video_path_proceed()
