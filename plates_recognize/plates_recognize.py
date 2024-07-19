import math

from ultralytics import YOLO

import os
import cv2
import numpy as np
import pytesseract

from cam_video_parse import image_show

# Загрузка обученной модели
# model = YOLO("best.pt")
model = YOLO("best-obb.pt")


def plates_get(predict, show=True):
    # для обычного bb
    x1, y1, x2, y2 = predict.boxes.xyxy.tolist()[0]
    selected_area = predict.orig_img[int(y1):int(y2), int(x1):int(x2)]
    if show:
        image_show(selected_area)
    return selected_area


def adjust_brightness_contrast(image, target_brightness=70, target_contrast=40, delta=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_contrast = np.std(gray)

    if abs(mean_brightness - target_brightness) > delta or abs(std_contrast - target_contrast) > delta:
        alpha = target_brightness / mean_brightness
        beta = target_contrast / std_contrast
        adjusted_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        return adjusted_gray
    else:
        return gray


def contours_get(image_in, min_width_percent=7, max_width_percent=25, min_height_percent=30):
    height, width = image_in.shape[:2]
    gray = adjust_brightness_contrast(image_in)

    # Применение порогового преобразования
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Нахождение контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # оставляем контуры, размеры которых попадают в допустимые пределы
        if w < width * max_width_percent / 100 and w > width * min_width_percent / 100 and h > height * min_height_percent / 100:
            selected_contours.append(contour)

    print(f'Найдено {len(selected_contours)} контуров')

    return selected_contours

def img_resize(image_in, scale_factor):
    height, width = image_in.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = cv2.resize(image_in, (new_width, new_height))

    return resized_image

def symbols_get(image_in, crop_h=5, crop_w=3, scale_factor=3, show=True):
    symbols = []
    image_in = img_resize(image_in, scale_factor=scale_factor)

    height, width = image_in.shape[:2]
    crop_h = crop_h*scale_factor
    crop_w = crop_w*scale_factor
    # image_in = image_in[crop_h:height - crop_h, crop_w:width - crop_w]
    contours = contours_get(image_in)
    min_symbols_qty = 5  # минимальное кол-во символов которое должно быть на табличке
    if len(contours) < min_symbols_qty:
        print(' подрезаем исходное изображение')
        image_in = image_in[crop_h:height - crop_h, crop_w:width - crop_w]
        contours = contours_get(image_in)

    # Извлечение отдельных символов
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        symbol = image_in[y:y + h, x:x + w]
        symbols.append(symbol)
        if show:
            cv2.rectangle(image_in, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Рисование bounding box

        cv2.imshow('Symbols', image_in)
    cv2.waitKey(0)

    return symbols


def obb_get(predict, rotate=True, show=True):
    obb = predict.obb
    # predict.plot()
    # predict.show()
    x1, y1, x2, y2 = obb.xyxy.tolist()[0]
    obb_img = predict.orig_img[int(y1):int(y2), int(x1):int(x2)]
    height, width = obb_img.shape[:2]
    # print(f'{obb.xywhr=}')
    if rotate:
        angle_rad = obb.xywhr.tolist()[0][4]
        angle = math.degrees(angle_rad)
        if angle > 90:
            angle = (180 - angle) * -1

        # Создание матрицы преобразования для поворота изображения
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # Поворот изображения
        obb_img_out = cv2.warpAffine(obb_img, rotation_matrix, (width, height))
    else:
        obb_img_out = obb_img

    if show:
        cv2.imshow('Original Image', obb_img)
        cv2.imshow('Rotated Image', obb_img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return obb_img_out

def process_images_in_directory(directory_path=None, file_ext = ".jpg" ):
    if directory_path is None:
        directory_path = os.getcwd()
    for filename in os.listdir(directory_path):
        if filename.endswith(file_ext):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            result = model.predict(image)[0]
            plate_img = obb_get(result, show=False)
            symbols_get(plate_img)

if __name__ == '__main__':
    path=r'F:\_Programming\UII\car_detect\_no_doubles'
    process_images_in_directory(path)
