import cv2
import os
import numpy as np

previous_img = None
current_img = None
all_points = []
current_points = []

image_directory = r'C:\Fontes\triedsohard\data\cube'
image_files = sorted(os.listdir(image_directory))

cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
print(image_files[0])
first_image = cv2.imread(os.path.join(image_directory, image_files[0]))
height, width = first_image.shape[:2]
cv2.resizeWindow("Images", width * 2 , height)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x - width, y))
        cv2.circle(current_img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(current_img, str(len(current_points)), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Images", current_img)

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

    while True: #Todo: Contar quantidades de pontos na img1 e usar mesma qtde para as demais
        key = cv2.waitKey(1)
        if key == 27:  # 27 ASCII Esc
            break
        elif key == 32:  # 32 ASCII espaço
            if len(current_points) == 0:
                print("Clique nos pontos antes de pressionar a barra de espaço.")
            else:
                break

    print(f"Pontos da imagem {image_file}: {current_points}")
    all_points.append(current_points)
    current_points = []

pontos_medios = np.array([np.mean(linha, axis=0) for linha in all_points])
pontos_medios = [tuple(ponto) for ponto in pontos_medios]

print(pontos_medios)

all_points = np.array(all_points)
print("\nall_points: {}x{};".format(all_points.shape[0], all_points.shape[1]))

matriz_diferencas = np.zeros_like(all_points)
for i, linha in enumerate(all_points):
    for j, ponto in enumerate(linha):
        diff_x = ponto[0] - pontos_medios[i][0]
        diff_y = ponto[1] - pontos_medios[i][1]
        matriz_diferencas[i][j] = (diff_x, diff_y)

coordenadas_x = [[ponto[0] for ponto in linha] for linha in matriz_diferencas]
coordenadas_y = [[ponto[1] for ponto in linha] for linha in matriz_diferencas]
matriz_separada = coordenadas_x + coordenadas_y

np.savetxt("matriz_entrada_tomasi_kanade.txt", matriz_separada, fmt='%.2f')

matriz_diferencas = np.array(matriz_diferencas)
print("\nmatriz_diferencas: {}x{};".format(matriz_diferencas.shape[0], matriz_diferencas.shape[1]))

matriz_separada = np.array(matriz_separada)
print("\nmatriz_separada: {}x{};".format(matriz_separada.shape[0], matriz_separada.shape[1]))

U, S, Vt = np.linalg.svd(matriz_separada)
S = np.diag(S) #Criar matriz sigma
print("\nU: {}x{}; S: {}x{}; Vt: {}x{};".format(U.shape[0], U.shape[1], S.shape[0], S.shape[1], Vt.shape[0], Vt.shape[1]))

U1 = U[:,:3]
S1 = S[:3,:3]
sqrt_S1 = np.sqrt(S1)
Vt1 = Vt[:3,:]
print("\nU1: {}x{}; S1: {}x{}; Vt1: {}x{};".format(U1.shape[0], U1.shape[1], S1.shape[0], S1.shape[1], Vt1.shape[0], Vt1.shape[1]))

motion = np.dot(U1, sqrt_S1)
motion_line = motion[:,:1]
#print(f"motion_line : {motion_line}")
#print(type(motion_line))
middle = len(motion_line) // 2
iValues = motion_line[:middle] 
jValues = motion_line[middle:]

# it * Q * Qt * i = 1
# jt * Q * Qt * j = 1
# it * Q * Qt * j = 0



# structure = np.dot(sqrt_Sigma, Vt)
# structure = np.transpose(structure[:3])

# M, N = structure.shape
# np.savetxt('structure.xyz', structure, fmt='%.2f', delimiter=' ', header=f"{M}\nStructure\n", comments='')

cv2.destroyAllWindows()