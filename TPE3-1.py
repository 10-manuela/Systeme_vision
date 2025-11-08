import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

def preprocess_image(image):
    """
    Applique un prétraitement complet sur l'image
    """
    image="C://Users/PC/Documents/Système_vision//img1.jpg"
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Amélioration du contraste avec CLAHE (meilleur que equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # 2. Réduction du bruit avec filtre médian
    denoised = cv2.medianBlur(contrast_enhanced, 3)
    
    # 3. Binarisation adaptative
    binary_adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # 4. Binarisation Otsu (pour comparaison)
    _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Redressement (deskew) automatique
    try:
        # Créer une image binaire pour la détection d'angle
        _, binary_for_angle = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary_for_angle > 0))
        
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Appliquer la rotation
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(binary_otsu, M, (w, h), flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed = binary_otsu
            angle = 0
    except Exception as e:
        print(f"Redressement non appliqué : {e}")
        deskewed = binary_otsu
        angle = 0
    
    return {
        'gray': gray,
        'contrast_enhanced': contrast_enhanced,
        'denoised': denoised,
        'binary_adaptive': binary_adaptive,
        'binary_otsu': binary_otsu,
        'deskewed': deskewed,
        'angle_correction': angle
    }

def perform_ocr(image, lang='fra+eng', config='--psm 6'):
    """
    Effectue l'OCR sur une image avec configuration spécifique
    """
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        # Obtenir les données détaillées
        details = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        return text, details
    except Exception as e:
        print(f"Erreur OCR : {e}")
        return "", {}

def calculate_ocr_confidence(details):
    """
    Calcule la confiance moyenne de l'OCR
    """
    if not details or 'conf' not in details:
        return 0
    
    confidences = [float(conf) for conf in details['conf'] if float(conf) > 0]
    return np.mean(confidences) if confidences else 0

def compare_results(text_brut, text_pretraite, details_brut, details_pretraite):
    """
    Compare quantitativement les résultats OCR
    """
    print("\n" + "="*70)
    print("COMPARAISON QUANTITATIVE DES RÉSULTATS OCR")
    print("="*70)
    
    # Statistiques de base
    stats_brut = {
        'caracteres': len(text_brut),
        'mots': len(text_brut.split()),
        'lignes': len([l for l in text_brut.split('\n') if l.strip()]),
        'confiance': calculate_ocr_confidence(details_brut)
    }
    
    stats_pretraite = {
        'caracteres': len(text_pretraite),
        'mots': len(text_pretraite.split()),
        'lignes': len([l for l in text_pretraite.split('\n') if l.strip()]),
        'confiance': calculate_ocr_confidence(details_pretraite)
    }
    
    print(f"{'Métrique':<15} {'Image brute':<12} {'Image prétraitée':<15} {'Amélioration':<12}")
    print("-" * 60)
    
    for metric in ['caracteres', 'mots', 'lignes', 'confiance']:
        brut = stats_brut[metric]
        pretraite = stats_pretraite[metric]
        amelioration = pretraite - brut
        
        if metric == 'confiance':
            print(f"{metric:<15} {brut:<12.1f}% {pretraite:<15.1f}% {amelioration:<12.1f}%")
        else:
            print(f"{metric:<15} {brut:<12} {pretraite:<15} {amelioration:<12}")

def main():
    # --- Chargement de l'image d'archive ---
    image_path = "C:/Users/PC/Desktop/Bob/archive.jpeg"  # À adapter
    
    if not os.path.exists(image_path):
        print(f"Erreur : Fichier {image_path} non trouvé")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("Erreur : Impossible de charger l'image")
        return
    
    print(f"📄 Image chargée : {img.shape[1]}x{img.shape[0]} pixels")
    
    # --- OCR sur image brute ---
    print("\n🔍 Application de l'OCR sur l'image brute...")
    text_brut, details_brut = perform_ocr(img)
    
    # --- Prétraitement de l'image ---
    print("\n🔄 Prétraitement de l'image...")
    processed_images = preprocess_image(img)
    print(f"📐 Angle de redressement appliqué : {processed_images['angle_correction']:.2f}°")
    
    # --- OCR sur image prétraitée ---
    print("\n🔍 Application de l'OCR sur l'image prétraitée...")
    text_pretraite, details_pretraite = perform_ocr(processed_images['deskewed'])
    
    # --- Affichage des résultats textuels ---
    print("\n" + "="*70)
    print("RÉSULTATS OCR - IMAGE BRUTE")
    print("="*70)
    print(text_brut[:1000] if text_brut else "Aucun texte détecté")
    
    print("\n" + "="*70)
    print("RÉSULTATS OCR - IMAGE PRÉTRAITÉE")
    print("="*70)
    print(text_pretraite[:1000] if text_pretraite else "Aucun texte détecté")
    
    # --- Comparaison quantitative ---
    compare_results(text_brut, text_pretraite, details_brut, details_pretraite)
    
    # --- Visualisation comparative ---
    print("\n📊 Génération des visualisations...")
    
    # Configuration de l'affichage
    plt.figure(figsize=(20, 12))
    
    # Images originales et étapes de prétraitement
    images_to_show = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Image originale (RGB)"),
        (processed_images['gray'], "Niveaux de gris", 'gray'),
        (processed_images['contrast_enhanced'], "Contraste amélioré (CLAHE)", 'gray'),
        (processed_images['denoised'], "Bruit réduit (Filtre médian)", 'gray'),
        (processed_images['binary_otsu'], "Binarisation (Otsu)", 'gray'),
        (processed_images['deskewed'], f"Image prétraitée finale\n(Redressée: {processed_images['angle_correction']:.1f}°)", 'gray')
    ]
    
    for i, (image, title, *cmap) in enumerate(images_to_show, 1):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap=cmap[0] if cmap else None)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- Sauvegarde des résultats ---
    output_file = "C:/Users/PC/Desktop/Bob/resultats_ocr_comparaison.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TPE 3.1 - COMPARAISON OCR: IMAGE BRUTE vs IMAGE PRÉTRAITÉE\n")
            f.write("="*60 + "\n\n")
            
            f.write("RÉSULTAT IMAGE BRUTE:\n")
            f.write("-" * 30 + "\n")
            f.write(text_brut)
            f.write("\n\n")
            
            f.write("RÉSULTAT IMAGE PRÉTRAITÉE:\n")
            f.write("-" * 30 + "\n")
            f.write(text_pretraite)
            f.write("\n\n")
            
            f.write("STATISTIQUES DE COMPARAISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Caractères (brut/prétraité): {len(text_brut)} / {len(text_pretraite)}\n")
            f.write(f"Mots (brut/prétraité): {len(text_brut.split())} / {len(text_pretraite.split())}\n")
            f.write(f"Confiance OCR (brut/prétraité): {calculate_ocr_confidence(details_brut):.1f}% / {calculate_ocr_confidence(details_pretraite):.1f}%\n")
        
        print(f"\n💾 Résultats sauvegardés dans : {output_file}")
        
    except Exception as e:
        print(f"⚠ Erreur lors de la sauvegarde : {e}")
main()
