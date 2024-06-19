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
    '248': (0.15, 0.45, 0.55, 0.25),
    '249': (0.25, 0.0, 0.45, 0.35),
    '252': (0.35, 0.4, 0.45, 0.2)
}  # ключ - номер камеры. значения - кортеж координат области детекции
# x_percent, y_percent, w_percent, h_percent


CLASSES_TO_DETECT = [2, 3, 7]  # ищем машины, мотоциклы, грузовики

FRAMES_SKIP = 4  # пропуск части кадров в видео для ускорения работы


def get_cam_id(filename):
    parts = filename.split('.')
    parts = parts[-2].split('_')
    cam_num = parts[0]
    cam_numbers = list(CAM_AREAS.keys())

    return cam_num if cam_num in cam_numbers else cam_numbers[0]


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
            try:
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
            except Exception as e:
                print(f'Произошла ошибка при обработке кадра {frame_counter}: {str(e)}')
                continue  # Пропуск битого кадра и продолжение выполнения

    cap.release()
    cv2.destroyAllWindows()

    print(f'Сохранено {detection_counter} файлов в {RESULT_DIR}')


def video_path_proceed(show_img=False, files_extention='mp4'):
    video_path = os.path.join(CURRENT_DIR, SOURCE_DIR)
    if not os.path.exists(video_path):
        print(f'Путь {video_path} к обрабатываемым файлам не найден')
        return

    files_list = [file for file in os.listdir(video_path) if file.endswith(files_extention)]

    for file in files_list:
        video_proceed(file, cam_id=get_cam_id(file), show_img=show_img)


if __name__ == '__main__':
    # для тестирования рекомендуется попробовать на небольшом фрагменте.
    # раскомментировать следующие две строки, ввести имя файла

    # video_file_name = '10.121.15.252_01_20240527062249177_4-.mp4'
    # video_proceed(video_file_name, cam_id=get_cam_id(video_file_name), show_img=True)

    # следующую строку раскомментировать когда уже всё настроено. для обработки всех файлов в папке

    video_path_proceed()
