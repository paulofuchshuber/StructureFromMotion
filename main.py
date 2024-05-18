import cv2
import os
import numpy as np
from scipy.optimize import newton
from scipy.optimize import least_squares
from scipy.linalg import cholesky
import open3d as o3d

previous_img = None
current_img = None
all_points = []
current_points = []

image_directory = r'C:\Fontes\StructureFromMotion\data\cube'
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

all_points = np.array(all_points)
# print('---------------------------------------------------------------')
# print(all_points)
# print('---------------------------------------------------------------')
# print("\nall_points: {}x{};".format(W.shape[0], W.shape[1]))
# all_points = np.array([[[322, 337], [443, 319], [323, 477], [436, 453], [253, 374], [250, 261], [346, 250]],
# [[310, 333], [436, 320], [313, 475], [430, 457], [260, 372], [255, 259], [356, 250]],
# [[306, 335], [434, 321], [307, 475], [426, 457], [262, 371], [258, 259], [359, 251]],
# [[299, 333], [427, 324], [300, 474], [423, 461], [266, 372], [264, 257], [363, 252]],
# [[294, 331], [424, 324], [298, 472], [420, 461], [270, 368], [268, 256], [369, 251]]])

coordenadas_x = all_points[:,:,0]
coordenadas_y = all_points[:,:,1]

nFrames = all_points.shape[0]
nPoints = all_points.shape[1]

print('nFrames: ', nFrames, 'nPoints: ', nPoints)

all_points = np.zeros((2*nFrames, nPoints))
all_points[:nFrames, :] = coordenadas_x
all_points[nFrames:2 * nFrames, :] = coordenadas_y

matriz_separada = all_points - np.mean(all_points, axis=1)[:, None]
matriz_separada = matriz_separada.astype('float32')

#print(w_bar)
U, S, Vt = np.linalg.svd(matriz_separada, full_matrices=False)
S = np.diag(S)[:3, :3]
U = U[:, :3]
Vt = Vt[:3, :]

structure = np.dot(np.sqrt(S), Vt)
motion = np.dot(U, np.sqrt(S))

motion_i = motion[0:nFrames, :]
motion_j = motion[nFrames:2 * nFrames, :]

A = np.zeros((2 * nFrames, 6))
for i in range(nFrames):
    A[2 * i, 0] = (motion_i[i, 0] ** 2) - (motion_j[i, 0] ** 2)
    A[2 * i, 1] = 2 * ((motion_i[i, 0] * motion_i[i, 1]) - (motion_j[i, 0] * motion_j[i, 1]))
    A[2 * i, 2] = 2 * ((motion_i[i, 0] * motion_i[i, 2]) - (motion_j[i, 0] * motion_j[i, 2]))
    A[2 * i, 3] = (motion_i[i, 1] ** 2) - (motion_j[i, 1] ** 2)
    A[2 * i, 5] = (motion_i[i, 2] ** 2) - (motion_j[i, 2] ** 2)
    A[2 * i, 4] = 2 * ((motion_i[i, 2] * motion_i[i, 1]) - (motion_j[i, 2] * motion_j[i, 1]))

    A[2 * i + 1, 0] = motion_i[i, 0] * motion_j[i, 0]
    A[2 * i + 1, 1] = motion_i[i, 1] * motion_j[i, 0] + motion_i[i, 0] * motion_j[i, 1]
    A[2 * i + 1, 2] = motion_i[i, 2] * motion_j[i, 0] + motion_i[i, 0] * motion_j[i, 2]
    A[2 * i + 1, 3] = motion_i[i, 1] * motion_j[i, 1]
    A[2 * i + 1, 4] = motion_i[i, 2] * motion_j[i, 1] + motion_i[i, 1] * motion_j[i, 2]
    A[2 * i + 1, 5] = motion_i[i, 2] * motion_j[i, 2]

U1, S1, V1 = np.linalg.svd(A, full_matrices=False)
v = np.transpose(V1)[:, -1]
QQt = np.zeros((3, 3))
print(v[0])
print(v[3])
print(v[5])

QQt[0, 0] = v[0]
QQt[1, 1] = v[3]
QQt[2, 2] = v[5]
QQt[0, 1] = v[1]
QQt[1, 0] = v[1]
QQt[0, 2] = v[2]
QQt[2, 0] = v[2]
QQt[2, 1] = v[4]
QQt[1, 2] = v[4]

Q = np.linalg.cholesky(QQt)

R = np.dot(motion, Q)

Qt = np.linalg.inv(Q)

S = np.dot(Qt, structure)

X = S[0, :]
Y = S[1, :]
Z = S[2, :]

pointcloud = np.zeros((X.shape[0], 3))
pointcloud[:,0] = X
pointcloud[:,1] = Y
pointcloud[:,2] = Z

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud)
o3d.io.write_point_cloud("pointcloud.ply", pcd)
pcd_load = o3d.io.read_point_cloud("pointcloud.ply")
o3d.visualization.draw_geometries([pcd_load])

cv2.destroyAllWindows()
