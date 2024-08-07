БИБЛИОТЕКА ДЛЯ НАРЕЗКИ ВИДЕО
Файл cam_video_parse.py
Данный код помогает искать на видео различные объекты при помощи yolov8 Найденные объекты сохраняются в файлах формата jpg

до начала работы скрипта необходимо указать параметры:

SOURCE_DIR - путь для поиска видео файлов RESULT_DIR - путь для сохранения файлов изображений (при отсутствии создаст автоматически)
CAM_AREAS - словарь, содержащий имя камеры в качестве ключа и кортеж из координат области детекции в качестве значений список координат имеет вид x_percent, y_percent, w_percent, h_percent - координата х у левого верхнего угла, а также ширина и высота области в процентах от размера изображения задается опытным путем, с учетом расположения камер
CLASSES_TO_DETECT - список номеров классов объектов yolov8 которые нужно обнаружить на видео
FRAMES_SKIP - для ускорения обработки и облегчения получившегося набора изображений, можно обнаруживать объекты не на всех кадрах, а пропускать какое-то количество

для запуска скрипта нужно выполнить функцию <i>video_path_proceed()</i> данная функция ищет в директории SOURCE_DIR файлы с нужным расширением (по умолчанию mp4) и сохраняет результат в RESULT_DIR
также можно указать конкретный файл <i>video_proceed(video_filename)</i>
после обработки каждого видеофайла в консоль выводится количество сохраненных изображений и затраченное время

для реализации данного метода можно запустить детектирование объектов с параметром predictions = model.predict(save_txt=True) координаты bounding_box сохраняются в текстовом файле, далее можно их обработать

---

upd v2.0

ДОБАВЛЕНА ПОСТ ОБРАБОТКА ФАЙЛОВ
Ранее после обработки видео получались полные кадры. теперь мы обрабатываем их и обрезаем по bounding box 

Для обработки одного файла применяем функцию <i>image_proceed()</i>

show=True - удобно использовать с данным параметром при настройке зоны обнаружения. 
Показывает синим цветом зону обнаружения
bounding box отрисовываются стандартными средствами yolo. по умолчанию показывает имя класса и вероятность его детекции
Если bounding box левым нижним углом попадает в зону обнаружения - он отмечается зеленой точкой. Иначе точка красная
save=True - по умолчанию в случае обнаружения объекта в указанной зоне (с помощью is_in_detect_zone()), изображение объекта, обрезанное по bounding box, сохраняется.
Для настройки зоны обнаружения сохранение можно отключить, save=False

Для обработки изображений в папке используется <i>images_path_proceed()</i> с указание пути 
Обработанные сохраняются в POST_RESULT_DIR 

Реализована проверка движения объекта при обработке видео is_obj_motion()
Добавлен параметр MOTION_TRES. Это минимальное расстояние, на которое должен переместиться bounding box между соседними кадрами, чтобы автомобиль не считался неподвижным.
Расстояние измеряется в процентах от длины диагонали изображения, 1=100%
Кадр из видео сохраняется в том случае, когда на нем обнаружен целевой объект и когда он движется 
 
ОБНОВЛЕНЫ ПОЛИГОНЫ ДЛЯ КАМЕР
зоны обнаружения уменьшены. старый вариант закомментировал
оставлено для совместимости с ранее нарезанными изображениями
в новых разработках можно сразу сохранять уменьшенные изображения

ДОБАВЛЕНА БИБЛИОТЕКА ДЕДУБЛИКАТОР
Файл doubles_check.py
Используется для обработки _обрезанных_ изображений.  
Использует модель ResNet50 для поиска уникальных признаков изображений
Далее сортирует изображения по дате создания и сравнивает их _попарно_ i с i+1 
Такой метод применен, поскольку мы ранее получили изображения из последовательных кадров видео
Далее производится вычисление косинусного расстояния между векторами признаков, на основе чего находится вероятность того, что объекты на двух изображениях одинаковые
1 - 100% одинаковые изображения


Запуск дедубликатора функцией <i>compare_files_recursive()</i>
Задаются пути для исходных файлов, для сохранения результата и значение порога идентичности в % (1=100%)
Рекурсивно обходит все вложенные папки
Первое изображение в папке сохраняется, потом сравнивается со вторым. 
Если изображения одинаковы с вероятностью выше установленного порога - второе также сохраняется, далее переход на следующий шаг.
