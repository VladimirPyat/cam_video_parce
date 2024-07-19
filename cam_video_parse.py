import os
import csv
import time
from math import sqrt


def make_csv_log(result_dict, log_name='log.csv'):
    if os.path.exists(log_name):
        mode = 'a'  # Дописываем в существующий файл
    else:
        mode = 'w'  # Создаем новый файл

    with open(log_name, mode, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())
        if mode == 'w':
            writer.writeheader()

        writer.writerow(result_dict)


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
POST_RESULT_DIR = '_post_process'

# ключ - номер камеры. значения - кортеж координат области детекции
# x_percent, y_percent, w_percent, h_percent
# первая версия
# CAM_AREAS = {
#     '248': (0.15, 0.45, 0.55, 0.25),
#     '249': (0.25, 0.0, 0.45, 0.35),
#     '252': (0.35, 0.4, 0.45, 0.2)

# вторая версия - уменьшена зона детекции
CAM_AREAS = {
    '248': (0.20, 0.58, 0.36, 0.13),
    '249': (0.23, 0.18, 0.33, 0.08),
    '252': (0.3, 0.5, 0.28, 0.12)
}


CLASSES_TO_DETECT = [2, 3, 7]  # ищем машины, мотоциклы, грузовики

FRAMES_SKIP = 5  # пропуск части кадров в видео для ускорения работы

MOTION_TRES = 0.005  # порог изменения координат BB в % для определения движения объекта

_last_bb = [0, 0]


def get_cam_id(img_filename):
    parts = img_filename.split('.')
    parts = parts[3].split('_')
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
    x_percent, y_percent, w_percent, h_percent = CAM_AREAS[cam_id]
    x, y, w, h = int(width * x_percent), int(height * y_percent), int(width * w_percent), int(height * h_percent)
    selected_area = image[y:y + h, x:x + w]

    return selected_area


def save_crop(image_out, result_path, image_filename_out):
    frame_path = os.path.join(result_path, image_filename_out)
    print(f'{image_filename_out} cохранено  в {result_path}')
    cv2.imwrite(frame_path, image_out)


def image_show(image_in, label='Object'):
    cv2.imshow(label, image_in)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_in_detect_zone(xyxy_box, xyxy_zone):
    x11, _, __, y21 = xyxy_box
    x12, y12, x22, y22 = xyxy_zone

    if (x12 <= x11 <= x22) and (y12 <= y21 <= y22):
        return True

    return False


def result_proceed(result, cam_id, image_filename, show=True, save=True):
    saves_count = 0
    result_path = os.path.join(CURRENT_DIR, POST_RESULT_DIR)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # получаем изображение с отрисоваными боксами
    image_in = result.plot()
    height, width, _ = image_in.shape
    bb_xyxy_list = result.boxes.xyxyn.tolist()

    # отрисовываем зону детекции
    x_percent, y_percent, w_percent, h_percent = CAM_AREAS[cam_id]
    x, y, w, h = int(width * x_percent), int(height * y_percent), int(width * w_percent), int(height * h_percent)
    zone_xyxy = [x_percent, y_percent, x_percent + w_percent, y_percent + h_percent]
    image = cv2.rectangle(image_in, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # помечаем левый нижний угол боксов
    for bb_xyxy in bb_xyxy_list:
        x1_percent, y1_percent, x2_percent, y2_percent = bb_xyxy
        x1, x2 = x1_percent * width, x2_percent * width
        y1, y2 = y1_percent * height, y2_percent * height
        circle_color = (0, 205, 30) if is_in_detect_zone(bb_xyxy, zone_xyxy) else (0, 10, 155)
        cv2.circle(image_in, (int(x1), int(y2)), 15, circle_color, -1)
        # сохраняем изображения попавшие в зону детекции
        if save:
            if is_in_detect_zone(bb_xyxy, zone_xyxy):
                image_filename_out = image_filename[:image_filename.rfind(".")] + "_crop.jpg"
                selected_area = result.orig_img[int(y1):int(y2), int(x1):int(x2)]
                saves_count += 1
                save_crop(selected_area, POST_RESULT_DIR, image_filename_out)

    if show:
        resized_image = cv2.resize(image, (800, 600))
        image_show(resized_image)

    return saves_count


def yolo_obj_detect(image_in):
    predictions = model.predict(image_in, conf=0.75, classes=CLASSES_TO_DETECT)

    return predictions


def is_obj_detected(result):
    return True if len(result.boxes) > 0 else False


def is_obj_motion(yolo_result):
    global _last_bb
    prev_x, prev_y, *_ = _last_bb  # сохраненные координаты последнего BB

    bb_xywh_list = yolo_result.boxes.xywhn.tolist()
    now_x, now_y, *_ = bb_xywh_list[0]  # координаты текущего BB из результатов
    _last_bb = bb_xywh_list[0][:2]  # обновляем координаты последнего BB

    distance = sqrt((prev_x - now_x) ** 2 + (prev_y - now_y) ** 2)  # вычисляем пройденное расстояние
    result = True if distance > MOTION_TRES else False  # если машина на видео двигается (изменение больше заданного процента) - True
    if not result:
        print('объект неподвижен')
    return result


def image_proceed(img_path, image_filename, cam_id=tuple(CAM_AREAS.keys())[0], show=True, save=True):
    saves_count = 0
    full_img_path = os.path.join(img_path, image_filename)
    if not os.path.exists(full_img_path):
        print(f'Файл {full_img_path} не найден')
        return
    image = cv2.imread(full_img_path)
    results = yolo_obj_detect(image)
    for result in results:
        saves_count += result_proceed(result, cam_id, image_filename, show=show, save=save)

    return saves_count


@timer
def video_proceed(video_filename, cam_id=tuple(CAM_AREAS.keys())[0], show_img=False, files_dir=None):
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
    img_size_counter = 0

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
                    frame_path = os.path.join(CURRENT_DIR, RESULT_DIR, files_dir, frame_filename)
                    if show_img:
                        cv2.imshow('Frame', frame_crop)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    cv2.imwrite(frame_path, frame)  # Сохранение кадра если обнаружен объект
                    img_size_counter += os.path.getsize(frame_path)  # размер в байтах

            except Exception as e:
                print(f'Произошла ошибка при обработке кадра {frame_counter}: {str(e)}')
                continue  # Пропуск битого кадра и продолжение выполнения

    cap.release()
    cv2.destroyAllWindows()

    print(f'Сохранено {detection_counter} файлов в {RESULT_DIR}/{files_dir}. Общий размер {img_size_counter} байт.')
    return {
        'images_qty': detection_counter,
        'total_size_mb': round(img_size_counter / (1024 * 1024), 2),
        'video_filename': video_filename
    }


def video_path_proceed(show_img=False, create_dir=True, files_extension='mp4', csv_log=True):
    video_path = os.path.join(CURRENT_DIR, SOURCE_DIR)
    if not os.path.exists(video_path):
        print(f'Путь {video_path} к обрабатываемым файлам не найден')
        return

    files_list = [file for file in os.listdir(video_path) if file.endswith(files_extension)]

    if not is_path_exist():
        print(f'Путь {SOURCE_DIR} к обрабатываемым файлам не найден')
        return

    for file in files_list:

        if create_dir:
            files_dir = file
            dir_path = os.path.join(CURRENT_DIR, RESULT_DIR, files_dir)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        else:
            files_dir = None

        result = video_proceed(file, cam_id=get_cam_id(file), show_img=show_img, files_dir=files_dir)
        if csv_log:
            make_csv_log(result)


def images_path_proceed(img_path, files_ext='jpg', show=False):
    images_path = os.path.join(CURRENT_DIR, img_path)
    if not os.path.exists(images_path):
        print(f'Путь {images_path} к обрабатываемым файлам не найден')
        return

    log_list=[]
    for root, dirs, files in os.walk(images_path):
        files_count = 0
        saves_count = 0

        for file in files:
            if file.endswith(files_ext):
                files_count += 1
                result = image_proceed(root, file, get_cam_id(file), show)
                saves_count += result

        log_strings = {
            'path': root,
            'files_count': files_count,
            'saves_count': saves_count
        }
        log_list.append(log_strings)
        make_csv_log(log_strings, log_name='img_path_log.csv' )

    total_files_count = sum(map(lambda x: x['files_count'], log_list))
    total_saves_count = sum(map(lambda x: x['saves_count'], log_list))
    print(f'Обработано файлов {total_files_count}')
    print(f'Сохранено файлов {total_saves_count}')


if __name__ == '__main__':
    # для тестирования рекомендуется попробовать на небольшом фрагменте.
    # раскомментировать следующие две строки, ввести имя файла

    # video_file_name = '10.121.15.249_01_test_2- — копия.mp4'
    # video_proceed(video_file_name, cam_id=get_cam_id(video_file_name))

    # следующую строку раскомментировать когда уже всё настроено. для обработки всех файлов в папке

    #video_path_proceed()

    #для подбора координат боксов можно тестировать на отдельных файлах. save=False чтоб не сохранять

    # filename = '10.121.15.248_01_202405310618038_12.mp4_18990.jpg'
    # image_proceed('', filename, cam_id=get_cam_id(filename), save=False)

    images_path_proceed('_result')
