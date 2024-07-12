import os
import shutil

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
from scipy.spatial.distance import cosine


# Загрузка предобученной модели ResNet50
def get_model(filename):
    if not os.path.exists(filename):
        base_model = ResNet50(weights='imagenet', include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        model.save(filename)
        print('model saved')
    else:
        model = load_model(model_filename)
        print('model loaded')

    return model


# Загрузка и предобработка изображений
def image_prepare(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_1d = np.expand_dims(img_arr, axis=0)
    return preprocess_input(img_1d)


def compare_img(img1, img2, model):
    # Извлечение признаков с помощью модели
    features1 = model.predict(img1).flatten()
    features2 = model.predict(img2).flatten()
    # Вычисление косинусного расстояния между векторами признаков
    return 1 - cosine(features1, features2)


def compare_files_recursive(source_path, result_path, model, sim_threshold, show=True):
    copied_files_count = {}  # Словарь для отслеживания количества скопированных файлов из каждой папки
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    total_count=0
    for root, dirs, files in os.walk(source_path):
        copied_files_count[root] = 0  # Инициализация счетчика для каждой папки
        files.sort(key=lambda x: os.path.getctime(os.path.join(root, x)))  # Сортировка файлов по времени создания

        if files:
            file1 = os.path.join(root, files[0])
            shutil.copy(file1, result_path)  # Копирование самого первого файла
            copied_files_count[root] += 1  # Увеличение счетчика для текущей папки

            for i in range(len(files) - 1):
                file1 = os.path.join(root, files[i])
                file2 = os.path.join(root, files[i + 1])
                img1_ready = image_prepare(file1)
                img2_ready = image_prepare(file2)
                similarity = compare_img(img1_ready, img2_ready, model)
                total_count+=1
                if show:
                    print(f'Одинаковые объекты в файлах {files[i]} {files[i + 1]} c вероятностью {similarity:.2f}')

                if similarity < sim_threshold:
                    shutil.copy(file2, result_path)
                    copied_files_count[root] += 1  # Увеличение счетчика для текущей папки

    total_copied = sum(copied_files_count.values())
    print(f"обработано {total_count} файлов")
    print(f"скопировано {total_copied} файлов")
    print(f"вероятность {sim_threshold} ")
    # Запись информации о количестве скопированных файлов из каждой папки в лог файл
    with open('doubles_log.txt', 'w') as log_file:
        log_file.write(f"обработано {total_count} файлов\n")
        for folder, count in copied_files_count.items():
            log_file.write(f"Из папки {folder} скопировано {count} файлов\n")
        log_file.write(f"вероятность {sim_threshold} ")


if __name__ == '__main__':
    model_filename = 'resnet50_model.h5'
    model_pt = get_model(model_filename)


    SOURCE_PATH = '_post_process'
    RESULT_PATH = '_no_doubles'

    sim_thresh = 0.85

    compare_files_recursive(SOURCE_PATH, RESULT_PATH, model_pt, sim_thresh)
