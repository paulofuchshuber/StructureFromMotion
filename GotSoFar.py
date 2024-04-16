import cv2
import os
import numpy as np

previous_img = None
current_img = None
current_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(current_img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(current_img, str(len(current_points)), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Images", current_img)

image_directory = r'C:\Fontes\triedsohard\data\cube'
image_files = sorted(os.listdir(image_directory))

cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
first_image = cv2.imread(os.path.join(image_directory, '000000.jpg'))
height, width = first_image.shape[:2]
#cv2.resizeWindow("Images", width * 2, height)

for image_file in image_files:
    if previous_img is not None:
        previous_img = current_img[:, width:]
    current_img = cv2.imread(os.path.join(image_directory, image_file))    

    for point in current_points:
        cv2.circle(current_img, point, 3, (0, 0, 255), -1)
        cv2.putText(current_img, str(current_points.index(current_points) + 1), (current_points[0] - 10, current_points[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if current_img is not None:
        if previous_img is None:
            previous_img = np.zeros_like(current_img)
        current_img = np.concatenate((previous_img, current_img), axis=1)
        cv2.imshow("Images", current_img)
        cv2.setMouseCallback("Images", click_event)

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # 27 ASCII Esc
            break
        elif key == 32:  # 32 ASCII espaço
            if len(current_points) == 0:
                print("Por favor, clique nos pontos antes de pressionar a barra de espaço.")
            else:
                break

    print(f"Pontos da imagem {image_file}: {current_points}")

    current_points = []

cv2.destroyAllWindows()