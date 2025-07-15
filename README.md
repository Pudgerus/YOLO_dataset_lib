# YOLO Dataset Utility Functions Documentation

Этот документ описывает функции для работы с YOLO-датасетами: обработка изображений, аннотаций, файловая система и статистика.

## 📁 Структура директорий

### `make_iamges_labels_paths(image_dst, labels_dst, type)`

Создаёт подкаталоги `type` внутри папок изображений и меток.

---

## 📄 Работа с файлами

### `ensure_dir(path)`

Создаёт директорию, если она не существует.

### `copy_file(src, dst)`

Копирует файл `src` в `dst`, создавая при необходимости папки.

### `move_file(src, dst)`

Перемещает файл `src` в `dst`, создавая при необходимости папки.

### `delete_file(path)`

Удаляет файл, если он существует.

### `get_filename(path)`

Возвращает имя файла с расширением.

### `get_file_stem(path)`

Возвращает имя файла без расширения.

---

## 🖼️ Обработка изображений

### `list_images(dir_path)`

Возвращает список изображений в директории (`.jpg`, `.jpeg`, `.png`).

### `image_exists(path)`

Проверяет, существует ли файл изображения.

### `get_image_size(image_path)`

Возвращает ширину и высоту изображения.

### `resize_image(image_path, output_path, size=(640, 640))`

Масштабирует изображение до заданного размера и сохраняет в новый путь.

### `is_image_corrupted(path)`

Проверяет, можно ли открыть изображение с помощью PIL или OpenCV.

---

## 🏷️ Работа с аннотациями YOLO

### `list_labels(dir_path)`

Возвращает список путей к `.txt` меткам в папке.

### `get_image_label_pairs(img_dir, label_dir)`

Создаёт пары (изображение, метка) по совпадающим именам файлов.

### `read_yolo_label(label_path)`

Считывает YOLO-метки: `[class_id, x_center, y_center, width, height]`.

### `write_yolo_label(label_path, labels)`

Записывает YOLO-метки в файл.

### `filter_labels_by_class(labels, keep_classes)`

Фильтрует метки, оставляя только указанные классы.

### `reindex_classes(labels, mapping)`

Переименовывает классы в метках по словарю соответствия.

### `is_yolo_label_file(filepath)`

Проверяет, соответствует ли файл формату YOLO (числа, пробелы, точки).

### `is_label_empty(label_path)`

Проверяет, пуст ли файл метки (нет строк с содержимым).

---

## 🧪 Работа с датасетами

### `shuffle_pairs(pairs)`

Перемешивает список пар (изображение, метка).

### `split_pairs(pairs, ratios)`

Делит пары на `train`, `val`, `test` по заданным долям (tuple из 3 float).

### `split_list(lst, ratios=None)`

Разбивает список на части. Если `ratios` не переданы, делит на 3 равные части. Работает с произвольным числом частей.

### `count_files(dir_path, ext="")`

Считает количество файлов в папке (по расширению, если указано).

### `count_classes(labels_dir)`

Возвращает количество аннотаций каждого класса из всех `.txt` файлов.

### `random_seed(seed=42)`

Устанавливает seed для `random` и `numpy` для воспроизводимости.

---

## 📌 Зависимости

* `os`
* `shutil`
* `random`
* `numpy`
* `cv2`
* `PIL`
* `typing`
* `pathlib`
