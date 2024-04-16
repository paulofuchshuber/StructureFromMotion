import cv2
import os
import numpy as np

# Função para lidar com o evento de clique do mouse
def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, str(len(points)), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Images", img)

# Diretório das imagens
image_directory = r'C:\Fontes\triedsohard\data\cube'
image_files = sorted(os.listdir(image_directory))

# Criar janela para exibir imagens
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Images", 1200, 600)

# Variável para guardar a imagem anterior e os pontos
previous_img = None
points = []

for image_file in image_files:
    # Leitura da imagem
    img = cv2.imread(os.path.join(image_directory, image_file))

    # Desenhar os pontos na imagem atual
    for point in points:
        cv2.circle(img, point, 3, (0, 0, 255), -1)
        cv2.putText(img, str(points.index(point) + 1), (point[0] - 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Lidando com eventos do mouse e desenho de pontos
    cv2.imshow("Images", img)
    cv2.setMouseCallback("Images", click_event)
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # 27 é o código ASCII para a tecla Esc
            break
        elif key == 32:  # 32 é o código ASCII para a barra de espaço
            if len(points) == 0:
                print("Por favor, clique nos pontos antes de pressionar a barra de espaço.")
            else:
                break

    # Imprimindo os pontos
    print(f"Pontos da imagem {image_file}: {points}")

    # Limpar a lista de pontos para a próxima imagem
    points = []

# Fechar a janela
cv2.destroyAllWindows()
