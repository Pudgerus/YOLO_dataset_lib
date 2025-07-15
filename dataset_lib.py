import os
import shutil
import glob
from typing import List, Tuple
import random
from PIL import Image
from pathlib import Path
from collections import defaultdict
import json

def make_iamges_labels_paths(image_dst: str, labels_dst: str, type: str):
    """
    Создаёт папки для изображений и меток с указанным типом (категорией).

    Args:
        image_dst (str): Путь к корневой папке с изображениями.
        labels_dst (str): Путь к корневой папке с метками.
        type (str): Имя подкаталога (например, 'train', 'val', 'test').
    """
    ensure_dir(os.path.join(image_dst, type))
    ensure_dir(os.path.join(labels_dst, type))

def list_images(dir_path: str) -> List[str]:
    """
    Возвращает список путей к изображениям в директории с расширениями jpg, jpeg, png.

    Args:
        dir_path (str): Путь к директории с изображениями.

    Returns:
        List[str]: Список путей к изображениям.
    """
    exts = ('.jpg', '.jpeg', '.png')
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(exts)]

def list_labels(dir_path: str) -> List[str]:
    """
    Возвращает список путей к .txt файлам (меткам) в директории.

    Args:
        dir_path (str): Путь к директории с метками.

    Returns:
        List[str]: Список путей к файлам меток.
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.txt')]

def get_image_label_pairs(img_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """
    Формирует пары (изображение, метка) по совпадению имени файла без расширения.

    Args:
        img_dir (str): Путь к папке с изображениями.
        label_dir (str): Путь к папке с метками.

    Returns:
        List[Tuple[str, str]]: Список кортежей с путями к изображениям и их меткам.
    """
    img_files = list_images(img_dir)
    label_files = list_labels(label_dir)
    label_map = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}
    pairs = []
    for img in img_files:
        stem = os.path.splitext(os.path.basename(img))[0]
        if stem in label_map:
            pairs.append((img, label_map[stem]))
    return pairs

def ensure_dir(path: str):
    """
    Создаёт директорию, если она не существует.

    Args:
        path (str): Путь к директории.
    """
    os.makedirs(path, exist_ok=True)

def copy_file(src: str, dst: str):
    """
    Копирует файл из src в dst, создавая необходимые директории.

    Args:
        src (str): Путь к исходному файлу.
        dst (str): Путь к целевому файлу.
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def move_file(src: str, dst: str):
    """
    Перемещает файл из src в dst, создавая необходимые директории.

    Args:
        src (str): Путь к исходному файлу.
        dst (str): Путь к целевому файлу.
    """
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)

def delete_file(path: str):
    """
    Удаляет файл, если он существует.

    Args:
        path (str): Путь к файлу.
    """
    if os.path.exists(path):
        os.remove(path)

def get_filename(path: str) -> str:
    """
    Возвращает имя файла с расширением из пути.

    Args:
        path (str): Путь к файлу.

    Returns:
        str: Имя файла с расширением.
    """
    return os.path.basename(path)

def get_file_stem(path: str) -> str:
    """
    Возвращает имя файла без расширения из пути.

    Args:
        path (str): Путь к файлу.

    Returns:
        str: Имя файла без расширения.
    """
    return os.path.splitext(os.path.basename(path))[0]

def read_yolo_label(label_path: str) -> list[list[float]]:
    """
    Считывает YOLO-аннотации из файла .txt и возвращает список списков с числами.

    Args:
        label_path (str): Путь к файлу с аннотациями.

    Returns:
        list[list[float]]: Список аннотаций, где каждая — [class_id, x_center, y_center, width, height].
    """
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            labels.append([float(p) for p in parts])
    return labels

def write_yolo_label(label_path: str, labels: list[list[float]]):
    """
    Записывает список YOLO-аннотаций в файл.

    Args:
        label_path (str): Путь к файлу для записи.
        labels (list[list[float]]): Список аннотаций, каждая — [class_id, x_center, y_center, width, height].
    """
    with open(label_path, 'w') as file:
        for label in labels:
            line = ' '.join(str(int(label[0])) if i == 0 else f"{label[i]:.6f}" for i in range(5))
            file.write(line + '\n')

def filter_labels_by_class(labels: list[list[float]], keep_classes: list[int]) -> list[list[float]]:
    """
    Фильтрует аннотации, оставляя только те, у которых класс входит в keep_classes.

    Args:
        labels (list[list[float]]): Список аннотаций.
        keep_classes (list[int]): Список индексов классов для сохранения.

    Returns:
        list[list[float]]: Отфильтрованный список аннотаций.
    """
    return [label for label in labels if int(label[0]) in keep_classes]

def reindex_classes(labels: list[list[float]], mapping: dict[int, int]) -> list[list[float]]:
    """
    Заменяет индексы классов в аннотациях согласно словарю mapping.

    Args:
        labels (list[list[float]]): Список аннотаций.
        mapping (dict[int, int]): Словарь замены {старый_индекс: новый_индекс}.

    Returns:
        list[list[float]]: Список аннотаций с обновлёнными индексами классов.
    """
    new_labels = []
    for label in labels:
        class_id = int(label[0])
        if class_id in mapping:
            new_label = [mapping[class_id]] + label[1:]
            new_labels.append(new_label)
        else:
            new_labels.append(label)
    return new_labels

def shuffle_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Перемешивает пары изображений и соответствующих аннотаций.

    Args:
        pairs (List[Tuple[str, str]]): Список пар (путь к изображению, путь к метке).

    Returns:
        List[Tuple[str, str]]: Перемешанный список пар.
    """
    random.shuffle(pairs)
    return pairs

def split_pairs(pairs: List[Tuple[str, str]], ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
    """
    Делит список пар на обучающую, валидационную и тестовую выборки согласно заданным пропорциям.

    Args:
        pairs (List[Tuple[str, str]]): Список пар (изображение, метка).
        ratios (Tuple[float, float, float]): Пропорции для train/val/test. Сумма должна быть равна 1.0.

    Returns:
        Tuple[List, List, List]: Три списка: обучающая, валидационная и тестовая выборки.
    """
    total = len(pairs)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]

def count_files(dir_path: str, ext: str = "") -> int:
    """
    Считает количество файлов в папке (с возможностью фильтрации по расширению).

    Args:
        dir_path (str): Путь к директории.
        ext (str, optional): Расширение файла (например, '.jpg'). По умолчанию учитываются все файлы.

    Returns:
        int: Количество найденных файлов.
    """
    return len([
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and (f.endswith(ext) if ext else True)
    ])

def is_yolo_label_file(filepath: str) -> bool:
    """
    Проверяет, является ли файл YOLO-форматной аннотацией (все строки состоят из цифр, пробелов и точек).

    Args:
        filepath (str): Путь к файлу аннотации.

    Returns:
        bool: True, если файл в формате YOLO, иначе False.
    """
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 5 or not all(p.replace('.', '', 1).isdigit() for p in parts):
                    return False
        return True
    except Exception:
        return False

def get_image_size(image_path: str) -> tuple[int, int]:
    """
    Возвращает размеры изображения (ширина, высота).

    Args:
        image_path (str): Путь к изображению.

    Returns:
        tuple[int, int]: (ширина, высота)
    """
    with Image.open(image_path) as img:
        return img.width, img.height

def resize_image(image_path: str, output_path: str, size=(640, 640)):
    """
    Масштабирует изображение до заданного размера и сохраняет по новому пути.

    Args:
        image_path (str): Путь к исходному изображению.
        output_path (str): Путь для сохранения изменённого изображения.
        size (tuple[int, int], optional): Размер (ширина, высота). По умолчанию (640, 640).
    """
    with Image.open(image_path) as img:
        resized = img.resize(size, Image.ANTIALIAS)
        resized.save(output_path)

def image_exists(path: str) -> bool:
    """
    Проверяет, существует ли файл изображения по указанному пути.

    Args:
        path (str): Путь к изображению.

    Returns:
        bool: True, если файл существует и это файл, иначе False.
    """
    return os.path.isfile(path)

import random
import numpy as np
from typing import List, Tuple, Dict

def split_list(lst: list, ratios: Tuple[float, ...] = None) -> Tuple[list, ...]:
    """
    Разбивает список на части по заданным долям.

    Если ratios не переданы, делит на 3 равные части.

    Args:
        lst (list): Исходный список.
        ratios (Tuple[float, ...], optional): Доли для разбиения, сумма должна быть 1.0.

    Returns:
        Tuple[list, ...]: Кортеж с разбитыми частями списка.
    """
    n = len(lst)
    if ratios is None:
        parts = 3
        ratios = tuple([1/parts] * parts)
    else:
        if not abs(sum(ratios) - 1.0) < 1e-6:
            raise ValueError("Сумма ratios должна быть равна 1.0")
    
    splits = []
    start = 0
    for ratio in ratios[:-1]:
        end = start + int(ratio * n)
        splits.append(lst[start:end])
        start = end
    splits.append(lst[start:]) 

    return tuple(splits)

def random_seed(seed=42):
    """
    Устанавливает seed для random и numpy для воспроизводимости.

    Args:
        seed (int, optional): Значение seed. По умолчанию 42.
    """
    random.seed(seed)
    np.random.seed(seed)

def count_classes(labels_dir: str) -> Dict[int, int]:
    """
    Считает количество аннотаций каждого класса в папке с .txt файлами.

    Args:
        labels_dir (str): Путь к папке с .txt аннотациями YOLO.

    Returns:
        Dict[int, int]: Словарь {class_id: count}
    """
    from pathlib import Path

    counts = {}
    path_obj = Path(labels_dir)
    for label_file in path_obj.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
    return counts

def is_label_empty(label_path: str) -> bool:
    """
    Проверяет, пустой ли файл аннотации (.txt).

    Args:
        label_path (str): Путь к файлу аннотации.

    Returns:
        bool: True если файл пустой или содержит только пустые строки, иначе False.
    """
    try:
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    return False
        return True
    except FileNotFoundError:
        return True

def is_image_corrupted(path: str) -> bool:
    """
    Проверяет, можно ли открыть изображение (PIL или OpenCV).

    Args:
        path (str): Путь к изображению.

    Returns:
        bool: True если изображение повреждено или не открывается, иначе False.
    """
    from PIL import Image
    import cv2

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return True

    try:
        img_cv = cv2.imread(path)
        if img_cv is None:
            return True
    except Exception:
        return True

    return False

def split_dataset(pairs: List[Tuple[str, str]], ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
    """
    Разбивает датасет (список пар изображение+метка) на train, val, test по заданным долям.

    Args:
        pairs (List[Tuple[str, str]]): Список пар (изображение, метка).
        ratios (Tuple[float, float, float]): Пропорции для train/val/test. Сумма должна быть 1.0.

    Returns:
        Tuple[List, List, List]: Разделённые части датасета.
    """
    from random import shuffle
    total = len(pairs)
    shuffle(pairs)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]

def copy_dataset_subset(pairs: List[Tuple[str, str]], image_dst: str, label_dst: str, subset: str):
    """
    Копирует изображения и метки в соответствующие подпапки (train/val/test).

    Args:
        pairs (List[Tuple[str, str]]): Список пар (изображение, метка).
        image_dst (str): Путь к папке для изображений.
        label_dst (str): Путь к папке для меток.
        subset (str): Название подкаталога (например, 'train').
    """
    for img, lbl in pairs:
        img_dst = os.path.join(image_dst, subset, os.path.basename(img))
        lbl_dst = os.path.join(label_dst, subset, os.path.basename(lbl))
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
        shutil.copy2(img, img_dst)
        shutil.copy2(lbl, lbl_dst)

def move_dataset_subset(pairs: List[Tuple[str, str]], image_dst: str, label_dst: str, subset: str):
    """
    Перемещает изображения и метки в соответствующие подпапки (train/val/test).

    Args:
        pairs (List[Tuple[str, str]]): Список пар (изображение, метка).
        image_dst (str): Путь к папке для изображений.
        label_dst (str): Путь к папке для меток.
        subset (str): Название подкаталога (например, 'val').
    """
    for img, lbl in pairs:
        img_dst = os.path.join(image_dst, subset, os.path.basename(img))
        lbl_dst = os.path.join(label_dst, subset, os.path.basename(lbl))
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
        shutil.move(img, img_dst)
        shutil.move(lbl, lbl_dst)

def reindex_dataset_classes(labels_dir: str, mapping: Dict[int, int]):
    """
    Изменяет индексы классов во всех метках папки по заданной карте.

    Args:
        labels_dir (str): Путь к папке с метками.
        mapping (Dict[int, int]): Словарь переиндексации {старый: новый}.
    """
    for file in Path(labels_dir).rglob("*.txt"):
        labels = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                new_id = mapping.get(class_id, class_id)
                labels.append([new_id] + [float(x) for x in parts[1:]])
        with open(file, 'w') as f:
            for label in labels:
                f.write(f"{int(label[0])} {' '.join(f'{x:.6f}' for x in label[1:])}\n")

def generate_class_statistics(labels_dir: str) -> Dict[int, int]:
    """
    Считает количество объектов каждого класса во всей папке меток.

    Args:
        labels_dir (str): Путь к папке с метками.

    Returns:
        Dict[int, int]: Статистика {class_id: количество объектов}.
    """
    stats = defaultdict(int)
    for file in Path(labels_dir).rglob("*.txt"):
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    stats[class_id] += 1
    return dict(stats)

def split_labels_by_class(labels_dir: str, output_dir: str):
    """
    Разносит аннотации по папкам классов: каждый файл содержит только метки одного класса.

    Args:
        labels_dir (str): Путь к папке с аннотациями.
        output_dir (str): Путь к выходной папке, где будут созданы подкаталоги по классам.
    """
    for file in Path(labels_dir).rglob("*.txt"):
        filename = file.stem
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = parts[0]
            class_dir = os.path.join(output_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            target_file = os.path.join(class_dir, filename + ".txt")
            with open(target_file, 'a') as out_f:
                out_f.write(line)

def backup_dataset(dataset_dir: str, backup_path: str):
    """
    Создаёт резервную копию папки датасета.

    Args:
        dataset_dir (str): Путь к исходной папке датасета.
        backup_path (str): Путь к резервной копии (архив или директория).
    """
    shutil.make_archive(backup_path, 'zip', dataset_dir)

def restore_dataset_backup(backup_path: str, restore_dir: str):
    """
    Восстанавливает датасет из архива.

    Args:
        backup_path (str): Путь к .zip архиву.
        restore_dir (str): Папка, в которую будет распакован архив.
    """
    shutil.unpack_archive(backup_path, restore_dir, 'zip')

def train_val_test_paths(image_src: str, image_dst: str, labels_src: str = '', labels_dst: str = '', train: bool = True, val: bool = True, test: bool = True):
    if labels_dst == '':
        labels_dst = image_dst.replace('images', 'labels')
    
    if labels_src == '':
        labels_src = image_src.replace('images', 'labels')
    
    if train:
        make_iamges_labels_paths(image_dst, labels_dst, 'train')
    
    if val:
        make_iamges_labels_paths(image_dst, labels_dst, 'val')
    
    if test:
        make_iamges_labels_paths(image_dst, labels_dst, 'test')