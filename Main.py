import cv2
import numpy as np
import os

list_cont = [[] for _ in range(42)]
list_x_y = [[] for _ in range(42)]
list_cont_2 = [[] for _ in range(42)]
list_x_y_2 = [[] for _ in range(42)]

hsv_ranges = [
    ((0, 100, 100), (8, 255, 255)),
    ((10, 100, 100), (25, 255, 255)),
    ((35, 100, 100), (50, 255, 255)),
    ((50, 50, 150), (70, 255, 255)),
    ((165, 100, 80), (179, 255, 200)),
    ((0, 50, 100), (10, 255, 255)),
    ((140, 80, 80), (160, 255, 255)),
    ((0, 30, 100), (10, 120, 220)),
    ((0, 30, 120), (10, 120, 255)),
    ((0, 100, 30), (10, 255, 100)),
    ((0, 10, 200), (179, 40, 255)),
    ((60, 30, 180), (90, 120, 255)),
    ((80, 80, 100), (100, 255, 255)),
    ((140, 50, 80), (160, 150, 200)),
    ((130, 30, 180), (160, 100, 255)),
    ((160, 80, 150), (179, 255, 255)),
    ((50, 20, 200), (70, 80, 255)),
    ((170, 80, 50), (179, 255, 150)),
    ((0, 0, 80), (179, 30, 140)),
    ((0, 0, 100), (179, 30, 160)),
    ((0, 0, 200), (179, 20, 255)),
    ((0, 0, 200), (179, 20, 255)),
    ((15, 50, 80), (30, 150, 200)),
    ((10, 50, 40), (30, 150, 120)),
    ((50, 30, 100), (70, 100, 200)),
    ((145, 100, 100), (170, 255, 255)),
    ((130, 30, 200), (160, 80, 255)),
    ((140, 80, 150), (160, 255, 255)),
    ((140, 100, 100), (160, 255, 255)),
    ((190, 80, 150), (210, 255, 255)),
    ((150, 80, 150), (170, 255, 255)),
    ((70, 80, 20), (90, 255, 100)),
    ((190, 50, 80), (220, 150, 200)),
    ((140, 100, 100), (160, 255, 255)),
    ((50, 30, 100), (70, 100, 200)),
    ((20, 30, 180), (40, 100, 255)),
    ((15, 50, 100), (30, 150, 200)),
    ((35, 30, 180), (55, 100, 255)),
    ((60, 80, 100), (90, 255, 255)),
    ((60, 80, 150), (90, 200, 255)),
    ((40, 80, 30), (80, 255, 150)),
    ((90, 10, 200), (120, 50, 255)),
]


def resize_image_for_display(img, max_display_width=1200, max_display_height=800):
    height, width = img.shape[:2]
    scale_x = max_display_width / width
    scale_y = max_display_height / height
    scale = min(scale_x, scale_y, 1.0)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Изображение сжато с {width}x{height} до {new_width}x{new_height}")
        return resized, scale
    else:
        print("Изображение не требует сжатия для отображения")
        return img, 1.0


def extract_largest_region_with_resize(image_path, output_path, max_display_size=1200, margin=100):
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Ошибка загрузки изображения!")
        return

    original_height, original_width = original_img.shape[:2]
    print(f"Оригинальный размер: {original_width}x{original_height}")

    display_img, scale = resize_image_for_display(original_img, max_display_size, max_display_size)

    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Контуры не найдены!")
        return

    largest_contour = max(contours, key=cv2.contourArea)

    if scale != 1.0:
        largest_contour = (largest_contour / scale).astype(np.int32)

    mask = np.zeros((original_height, original_width), dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(largest_contour)

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(original_width - x, w + 2 * margin)
    h = min(original_height - y, h + 2 * margin)

    result = original_img[y:y + h, x:x + w]

    cv2.imwrite(output_path, result)
    print(f"Результат сохранен: {output_path}")
    print(f"Размер вырезанной области: {w}x{h} пикселей")
    print(f"Использован отступ: {margin} пикселей")

    return result


def place(image, list_x_y, list_cont):
    for i in range(len(hsv_ranges)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_ranges[i][0], hsv_ranges[i][1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                list_x_y[i].append((cx, cy))
                list_cont[i].append(cnt)


def find_similar_contours(contours1, contours2, tolerance_percent=10, image_shape=None):
    similar_pairs = []
    different_contours1 = []
    different_contours2 = []

    if image_shape is not None:
        max_distance = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2) * tolerance_percent / 100
    else:
        max_distance = 50

    used_indices_2 = set()

    for i, cnt1 in enumerate(contours1):
        M1 = cv2.moments(cnt1)
        if M1["m00"] != 0:
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])

            found_similar = False
            for j, cnt2 in enumerate(contours2):
                if j in used_indices_2:
                    continue

                M2 = cv2.moments(cnt2)
                if M2["m00"] != 0:
                    cx2 = int(M2["m10"] / M2["m00"])
                    cy2 = int(M2["m01"] / M2["m00"])

                    distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                    if distance <= max_distance:
                        similar_pairs.append((cnt1, cnt2))
                        used_indices_2.add(j)
                        found_similar = True
                        break

            if not found_similar:
                different_contours1.append(cnt1)

    for j, cnt2 in enumerate(contours2):
        if j not in used_indices_2:
            different_contours2.append(cnt2)

    return similar_pairs, different_contours1, different_contours2


def create_highlighted_image(base_image, changed_contours):

    result_image = base_image.copy()
    bright_blue = (255, 0, 0)


    for cnt in changed_contours:
        area = cv2.contourArea(cnt)
        if area < 15000:
            mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            result_image[mask == 255] = bright_blue

    return result_image


def compare_all_categories(list_cont1, list_cont2, image_shape):

    all_changed_contours = []

    for category_idx in range(len(list_cont1)):
        contours1 = list_cont1[category_idx]
        contours2 = list_cont2[category_idx]

        print(f"Категория {category_idx}: {len(contours1)} контуров в 1-м, {len(contours2)} контуров во 2-м")

        similar_contours, different_contours1, different_contours2 = find_similar_contours(
            contours1, contours2, tolerance_percent=10, image_shape=image_shape
        )

        all_changed_contours.extend(different_contours1)
        all_changed_contours.extend(different_contours2)

        print(f"  -> {len(similar_contours)} похожих, "
              f"{len(different_contours1)} уникальных в 1-м, "
              f"{len(different_contours2)} уникальных во 2-м")

    return all_changed_contours


if __name__ == "__main__":
    res_1 = extract_largest_region_with_resize("image1.jpg", "result.jpg", margin=450)
    res_2 = extract_largest_region_with_resize("image2.jpg", "result2.jpg", margin=450)

    if res_1.shape[0] > res_2.shape[0]:
        res_1 = cv2.resize(res_1, (res_2.shape[1], res_2.shape[0]))
    else:
        res_2 = cv2.resize(res_2, (res_1.shape[1], res_1.shape[0]))

    result_display, _ = resize_image_for_display(res_1, 1200, 5000)
    result_display_2, _ = resize_image_for_display(res_2, 1200, 5000)

    print(f"Размер первого изображения: {res_1.shape}")
    print(f"Размер второго изображения: {res_2.shape}")

    place(result_display, list_x_y, list_cont)
    place(result_display_2, list_x_y_2, list_cont_2)

    all_changed_contours = compare_all_categories(list_cont, list_cont_2, result_display.shape)

    print(f"Всего измененных областей: {len(all_changed_contours)}")

    highlighted_image = create_highlighted_image(result_display, all_changed_contours)

    cv2.imwrite("changed_areas_blue.jpg", highlighted_image)
    print("Изображение с выделенными изменениями сохранено в файл: changed_areas_blue.jpg")

    cv2.imshow('Первое изображение', result_display)
    cv2.imshow('Второе изображение', result_display_2)
    cv2.imshow('Измененные области (ярко-синие)', highlighted_image)

    print("Нажмите любую клавишу чтобы закрыть окна...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()