import cv2
import mediapipe as mp

# Initialisation de MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Démarrer la capture vidéo depuis la webcam (0 est généralement la webcam par défaut)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Lire une image de la webcam
    success, image = cap.read()
    if not success:
        print("Ignorer l'image vide de la webcam.")
        continue

    # Convertir l'image de BGR (format OpenCV) à RGB (format MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Traiter l'image et détecter les visages
    results = face_detection.process(image_rgb)

    # Reconvertir l'image en BGR pour l'affichage OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Si des visages sont détectés, dessiner les boîtes
    if results.detections:
        for detection in results.detections:
            # Méthode 1 : Dessiner avec les outils de dessin de MediaPipe (le plus simple)
            mp_drawing.draw_detection(image_bgr, detection)

            # (Pour plus de contrôle plus tard, on peut aussi récupérer les coordonnées manuellement)
            # bboxC = detection.location_data.relative_bounding_box
            # h, w, c = image.shape
            # x = int(bboxC.xmin * w)
            # y = int(bboxC.ymin * h)
            # width = int(bboxC.width * w)
            # height = int(bboxC.height * h)
            # cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

    # Afficher l'image dans une fenêtre
    cv2.imshow('Etape 1 - Detection Visage (MediaPipe)', image_bgr)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()