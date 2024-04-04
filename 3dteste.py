#solvepnp (pose_estimation_implement.py) + cornerDetector.py(keypoints) + fundamental_mat_estimation (matching points)

import glob
import os
import cv2 as cv
import numpy as np

# Definir o diretório das imagens
padrao = r'C:\Fontes\myProject\data\cube\*.jpg'
arquivos_imagem = sorted(glob.glob(padrao))

f, cx, cy = 793.64835681, 818.82194152, 533.71057056 #ajustar
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

# Iterar sobre as imagens em pares
for pathImagem1, pathImagem2 in zip(arquivos_imagem[:-1], arquivos_imagem[1:]):
    # Extrair apenas o nome do arquivo sem a extensão
    nome_arquivo_sem_extensao1 = os.path.splitext(os.path.basename(pathImagem1))[0]
    nome_arquivo_sem_extensao2 = os.path.splitext(os.path.basename(pathImagem2))[0]
    
    # Exibir os nomes dos arquivos sem extensão em pares
    print("Imagem 1:", nome_arquivo_sem_extensao1)
    print("Imagem 2:", nome_arquivo_sem_extensao2)
    
    # Carregar as imagens usando OpenCV
    imagem1 = cv.imread(pathImagem1)
    imagem2 = cv.imread(pathImagem2)
    
    # Verificar se as imagens foram carregadas corretamente
    if imagem1 is not None and imagem2 is not None:
        # Exibir as imagens
        cv.imshow('Imagem 1', imagem1)
        cv.imshow('Imagem 2', imagem2)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        brisk = cv.BRISK_create()
        keypoints1, descriptors1 = brisk.detectAndCompute(imagem1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(imagem2, None)
        
        fmatcher = cv.DescriptorMatcher_create('BruteForce')
        match = fmatcher.match(descriptors1, descriptors2)
        
        pts1, pts2 = [], []
        for i in range(len(match)):
            pts1.append(keypoints1[match[i].queryIdx].pt)
            pts2.append(keypoints2[match[i].trainIdx].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        F, inlier_mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 0.5, 0.999)
        E = K.T @ F @ K
        positive_num, R, t, positive_mask = cv.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
        print(f'* The position of Image #2 = {-R.T @ t}') # [-0.57, 0.09, 0.82]
        img_matched = cv.drawMatches(imagem1, keypoints1, imagem2, keypoints2, match, None, None, None, matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
        cv.namedWindow('Fundamental Matrix Estimation', cv.WINDOW_NORMAL)
        cv.imshow('Fundamental Matrix Estimation', img_matched)





# corners:
# img = cv2.imread(diretorio_imagem)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray, 25, 0.1, 10)
# corners = np.intp(corners)
# for i in corners:
    # x, y = i.ravel()
    # cv2.circle(img, (x, y), 9, 255, -1)
# cv2.imshow('img', img)
# cv2.waitKey(0)