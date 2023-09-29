import cv2
import numpy as np
import os

def dHash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def calculate_hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

# Ввод пути к файлу изображения с клавиатуры
file_path = input("Введите путь к изображению: ")

# Проверка наличия файла
if os.path.exists(file_path):
    img = cv2.imread(file_path)
    image_output = img.copy()

    # конвертация в grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # пороговая бинаризация для выделения объектов
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # поиск контуров на бинаризированном изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # фильтрация контуров по размеру (площади) для выделения кругов
    min_radius = 16
    max_radius = 20
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_radius**2 * np.pi <= area <= max_radius**2 * np.pi:
            circles.append(contour)

    # рисование кругов на исходном изображении
    for circle in circles:
        (x, y), radius = cv2.minEnclosingCircle(circle)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image_output, center, radius, (0, 255, 0), 2)
        cv2.circle(image_output, center, 2, (0, 0, 255), 3)

    # Вывод всех кругов в консоль
    print("Количество найденных кругов:", len(circles))

    cv2.imshow("The result", image_output)
    cv2.waitKey(0)
else:
    print("Файл не найден. Пожалуйста, убедитесь, что указали правильный путь к изображению.")
