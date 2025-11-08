import cv2
import mediapipe as mp
import numpy as np

# Initialisation de MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

# Variable pour changer de filtre avec le clavier
current_filter = "flou"  # Commence avec le filtre flou

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convertir BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Détection des visages
    results = face_detection.process(image_rgb)

    # Reconvertir en BGR pour le traitement OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Faire une copie de l'image originale pour y appliquer les filtres localement
    filtered_image = image_bgr.copy()

    if results.detections:
        for detection in results.detections:
            # Récupérer les coordonnées de la boîte englobante
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = image_bgr.shape
            
            # Convertir les coordonnées relatives en coordonnées pixels
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            
            # S'assurer que les coordonnées sont dans les limites de l'image
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)
            
            # Extraire la région du visage
            face_roi = image_bgr[y:y+height, x:x+width]
            
            # Appliquer le filtre sélectionné sur la région du visage
            if face_roi.size > 0:  # Vérifier que la région n'est pas vide
                if current_filter == "flou":
                    # FILTRE 1: Flou gaussien fort
                    filtered_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    
                elif current_filter == "cartoon":
                    # FILTRE 2: Effet cartoon
                    # Étape 1: Réduire le bruit avec un flou médian
                    color = cv2.bilateralFilter(face_roi, 9, 300, 300)
                    
                    # Étape 2: Créer un effet de dessin au trait
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.medianBlur(gray, 7)
                    edges = cv2.adaptiveThreshold(gray_blur, 255, 
                                                cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                cv2.THRESH_BINARY, 9, 10)
                    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    
                    # Étape 3: Combiner couleur et contours
                    filtered_face = cv2.bitwise_and(color, edges_color)
                    
                elif current_filter == "negatif":
                    # FILTRE 3: Négatif des couleurs
                    filtered_face = cv2.bitwise_not(face_roi)
                    
                elif current_filter == "aucun":
                    # Pas de filtre - visage original
                    filtered_face = face_roi
                
                # Remplacer la région du visage originale par la version filtrée
                filtered_image[y:y+height, x:x+width] = filtered_face
                
                # Dessiner un rectangle vert pour montrer la zone de détection
                cv2.rectangle(filtered_image, (x, y), (x+width, y+height), (0, 255, 0), 2)

    # Ajouter du texte pour indiquer le filtre actuel
    cv2.putText(filtered_image, f"Filtre: {current_filter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(filtered_image, "Touches: 1=Flou, 2=Cartoon, 3=Negatif, 4=Aucun, q=Quitter", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Afficher l'image avec filtres
    cv2.imshow('Etape 2 - Filtres Visage', filtered_image)

    # Gestion des touches pour changer de filtre
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_filter = "flou"
    elif key == ord('2'):
        current_filter = "cartoon"
    elif key == ord('3'):
        current_filter = "negatif"
    elif key == ord('4'):
        current_filter = "aucun"

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()