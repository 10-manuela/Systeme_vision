import cv2
import mediapipe as mp
import numpy as np
import math

# Initialisation de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

# Variables pour suivre les états des expressions
eyes_closed = False
mouth_open = False
current_effect = "points"  # Effet par défaut

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convertir BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Reconvertir en BGR pour l'affichage
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        output_image = image_bgr.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = image_bgr.shape
                
                # Convertir les points de repère en coordonnées pixels
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                # DÉTECTION D'EXPRESSIONS
                # 1. Détection des yeux fermés (points 145, 159, 386, 374)
                left_eye_top = landmarks[159]    # Paupière supérieure gauche
                left_eye_bottom = landmarks[145] # Paupière inférieure gauche
                right_eye_top = landmarks[386]   # Paupière supérieure droite  
                right_eye_bottom = landmarks[374] # Paupière inférieure droite
                
                # Calcul de l'ouverture des yeux
                left_eye_open = abs(left_eye_top[1] - left_eye_bottom[1])
                right_eye_open = abs(right_eye_top[1] - right_eye_bottom[1])
                
                eyes_closed = (left_eye_open < 5) and (right_eye_open < 5)
                
                # 2. Détection de la bouche ouverte (points 13, 14, 78, 308)
                upper_lip = landmarks[13]    # Lèvre supérieure
                lower_lip = landmarks[14]    # Lèvre inférieure
                mouth_left = landmarks[78]   # Coin gauche bouche
                mouth_right = landmarks[308] # Coin droit bouche
                
                # Calcul de l'ouverture de la bouche
                mouth_open_height = abs(upper_lip[1] - lower_lip[1])
                mouth_width = abs(mouth_left[0] - mouth_right[0])
                mouth_open = (mouth_open_height > mouth_width * 0.3)
                
                # APPLIQUER LES EFFETS SELON L'EXPRESSION
                if current_effect == "points":
                    # Effet 1: Dessiner tous les points de repère
                    for landmark in landmarks:
                        cv2.circle(output_image, landmark, 1, (0, 255, 0), -1)
                    
                    # Dessiner les connections entre les points
                    mp_drawing.draw_landmarks(
                        image=output_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                
                elif current_effect == "yeux_effet":
                    # Effet 2: Effet spécial sur les yeux
                    # Points pour l'œil gauche (indices approximatifs)
                    left_eye_points = landmarks[33:42] + [landmarks[246], landmarks[161], landmarks[160]]
                    left_eye_contour = np.array(left_eye_points, dtype=np.int32)
                    
                    # Points pour l'œil droit
                    right_eye_points = landmarks[263:272] + [landmarks[466], landmarks[388], landmarks[387]]
                    right_eye_contour = np.array(right_eye_points, dtype=np.int32)
                    
                    # Appliquer un effet de couleur sur les yeux
                    if eyes_closed:
                        # Yeux fermés - effet de sommeil
                        cv2.fillPoly(output_image, [left_eye_contour], (0, 0, 255))  # Rouge
                        cv2.fillPoly(output_image, [right_eye_contour], (0, 0, 255)) # Rouge
                    else:
                        # Yeux ouverts - effet lumineux
                        cv2.fillPoly(output_image, [left_eye_contour], (255, 255, 0))  # Cyan
                        cv2.fillPoly(output_image, [right_eye_contour], (255, 255, 0)) # Cyan
                
                elif current_effect == "bouche_effet":
                    # Effet 3: Effet sur la bouche
                    mouth_points = landmarks[13:18] + landmarks[78:88] + landmarks[308:318]
                    mouth_contour = np.array(mouth_points, dtype=np.int32)
                    
                    if mouth_open:
                        # Bouche ouverte - effet cri
                        cv2.fillPoly(output_image, [mouth_contour], (0, 0, 255))  # Rouge
                        # Ajouter un texte "CRI!"
                        text_x = mouth_contour[:, 0].mean() - 20
                        text_y = mouth_contour[:, 1].mean() - 10
                        cv2.putText(output_image, "CRI!", (int(text_x), int(text_y)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        # Bouche fermée - effet normal
                        cv2.fillPoly(output_image, [mouth_contour], (0, 255, 0))  # Vert
                
                elif current_effect == "masque_texte":
                    # Effet 4: Masque avec texture
                    # Créer un masque pour tout le visage
                    face_oval_points = [
                        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
                        361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
                        176, 149, 150, 136, 172, 58, 132, 93, 234, 127
                    ]
                    face_contour = np.array([landmarks[i] for i in face_oval_points], dtype=np.int32)
                    
                    # Créer une texture (dégradé de couleurs)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [face_contour], 255)
                    
                    # Créer une texture colorée
                    texture = np.zeros((h, w, 3), dtype=np.uint8)
                    for i in range(h):
                        for j in range(w):
                            texture[i, j] = [j % 256, i % 256, (i+j) % 256]
                    
                    # Appliquer la texture seulement sur le masque du visage
                    output_image = cv2.bitwise_and(output_image, output_image, mask=~mask)
                    texture_masked = cv2.bitwise_and(texture, texture, mask=mask)
                    output_image = cv2.add(output_image, texture_masked)
        
        # Affichage des informations
        status_text = f"Effet: {current_effect}"
        cv2.putText(output_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        expression_text = f"Yeux: {'FERMES' if eyes_closed else 'ouverts'} - Bouche: {'OUVERTE' if mouth_open else 'fermee'}"
        cv2.putText(output_image, expression_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = "Touches: 1=Points, 2=Yeux, 3=Bouche, 4=Masque, q=Quitter"
        cv2.putText(output_image, instructions, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Afficher l'image
        cv2.imshow('Etape 3 - Points de Repère et Expressions', output_image)

        # Gestion des touches
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_effect = "points"
        elif key == ord('2'):
            current_effect = "yeux_effet"
        elif key == ord('3'):
            current_effect = "bouche_effet"
        elif key == ord('4'):
            current_effect = "masque_texte"

cap.release()
cv2.destroyAllWindows()