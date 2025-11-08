# TD 3 : Analyse d’archives numériques 
# Outils : Python, Tesseract OCR, OpenCV, pandas 
# Exercices : 
# 1. Extraire du texte d’une image de document manuscrit (OCR). 
import cv2
import pytesseract
import numpy as np
def ocr_manuscrit(image_path):
    img = cv2.imread("Moliere.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?'
    texte = pytesseract.image_to_string(thresh, config=config)
    print("=== TEXTE EXTRAIT ===")
    print(texte)
    cv2.imshow('Image originale', img)
    cv2.imshow('Image pretraitee', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return texte
texte_extrait = ocr_manuscrit('document_manuscrit.jpg')

# 2. Segmenter automatiquement des zones de texte et d’image.
def segmenter_document(image_path):
    img = cv2.imread("img.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobelx, sobely)
    _, mask_texte = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_texte.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultat = img.copy()
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1000:  # Filtrer les petites zones
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h
            if ratio > 2:  # Zone probablement texte
                cv2.rectangle(resultat, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(resultat, 'TEXTE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:  # Zone probablement image
                cv2.rectangle(resultat, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(resultat, 'IMAGE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('Segmentation', resultat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
segmenter_document('document_archive.jpg')

# 3. Identifier des motifs visuels récurrents (logos, symboles). 
def detecter_motifs(image_path, template_path):
    img = cv2.imread("image.png")
    template = cv2.imread("logo.png", 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    locations = np.where(res >= threshold)
    resultat = img.copy()
    for pt in zip(*locations[::-1]):
        cv2.rectangle(resultat, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        cv2.putText(resultat, 'LOGO', (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    print(f"Nombre de logos detectes: {len(locations[0])}")
    cv2.imshow('Detection Logos', resultat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
detecter_motifs('document_avec_logos.jpg', 'logo_template.jpg')

# 4. Créer une “carte visuelle” d’un corpus d’images patrimoniales. 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def creer_carte_visuelle(dossier_images):
    import os
    import glob
    # Charger toutes les images du dossier
    images_path = glob.glob(os.path.join(dossier_images, "*.jpg"))
    features = []
    noms_images = []
    for path in images_path:
        img = cv2.imread(path)
        img = cv2.resize(img, (100, 100))  # Redimensionner pour uniformité
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        features.append(hist)
        noms_images.append(os.path.basename(path))
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(features)
    plt.figure(figsize=(12, 8))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7)
    for i, nom in enumerate(noms_images):
        plt.annotate(nom, (coords_2d[i, 0], coords_2d[i, 1]), fontsize=8)
    plt.title('Carte Visuelle du Corpus d\'Archives')
    plt.xlabel('Composante PCA 1')
    plt.ylabel('Composante PCA 2')
    plt.grid(True, alpha=0.3)
    plt.show()
creer_carte_visuelle('corpus_archives/')