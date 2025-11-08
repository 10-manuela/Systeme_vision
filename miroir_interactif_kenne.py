import cv2
import numpy as np
import mediapipe as mp
import time
import os

# --------- CONFIG ----------
CAM_ID = 0
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Key mappings:
#   '1' = normal (no effect)
#   '2' = cartoon
#   '3' = glitch
#   '4' = deformation (yeux/sourire)
#   'm' = toggle mask overlay
#   's' = save snapshot
#   'q' or ESC = quit

# --------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# --------- helper filters ----------
def cartoonize(img):
    # bilateral filter + edge detection
    img_small = cv2.pyrDown(img)
    for _ in range(2):
        img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)
    img_up = cv2.pyrUp(img_small)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_up, edges)
    return cartoon

def glitch_effect(img):
    # simple RGB channel offsets + scanline noise
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    # shift channels by random small offsets
    shift = int(max(2, w * 0.01))
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=0)
    glitched = cv2.merge((b, g, r))
    # add horizontal scanlines
    overlay = glitched.copy()
    for y in range(0, h, 3):
        overlay[y:y+1, :] = (overlay[y:y+1, :] * 0.7).astype(np.uint8)
    glitched = cv2.addWeighted(glitched, 0.9, overlay, 0.1, 0)
    return glitched

def enlarge_roi(img, bbox, scale=1.3):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return img
    cx, cy = x + w // 2, y + h // 2
    # region coordinates padded
    pad_w, pad_h = int(w * 0.5), int(h * 0.6)
    x0 = max(0, cx - pad_w - w//2)
    y0 = max(0, cy - pad_h - h//2)
    x1 = min(img.shape[1], cx + pad_w + w//2)
    y1 = min(img.shape[0], cy + pad_h + h//2)
    roi = img[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return img
    new_w = int(roi.shape[1] * scale)
    new_h = int(roi.shape[0] * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # center paste
    cx2 = x0 + roi.shape[1] // 2
    cy2 = y0 + roi.shape[0] // 2
    x_start = max(0, cx2 - new_w // 2)
    y_start = max(0, cy2 - new_h // 2)
    x_end = min(img.shape[1], x_start + new_w)
    y_end = min(img.shape[0], y_start + new_h)
    # blend resized onto original
    region = img[y_start:y_end, x_start:x_end]
    # crop resized to fit if necessary
    cropped = resized[0:(y_end-y_start), 0:(x_end-x_start)]
    alpha = 0.9
    blended = cv2.addWeighted(cropped, alpha, region, 1 - alpha, 0)
    img[y_start:y_end, x_start:x_end] = blended
    return img

def mouth_aspect_ratio(landmarks, img_w, img_h):
    # use lip corner landmarks to estimate smile/height
    # landmarks are normalized (x,y)
    # left corner ~ 61, right corner ~291 (MediaPipe face mesh indices)
    # top lip ~ 13, bottom lip ~14 etc. We'll take some approximate indices
    try:
        left = landmarks[61]
        right = landmarks[291]
        top = landmarks[13]
        bottom = landmarks[14]
    except Exception:
        return 0
    left_pt = np.array([int(left.x * img_w), int(left.y * img_h)])
    right_pt = np.array([int(right.x * img_w), int(right.y * img_h)])
    top_pt = np.array([int(top.x * img_w), int(top.y * img_h)])
    bottom_pt = np.array([int(bottom.x * img_w), int(bottom.y * img_h)])
    horizontal = np.linalg.norm(left_pt - right_pt)
    vertical = np.linalg.norm(top_pt - bottom_pt)
    if horizontal == 0:
        return 0
    return vertical / horizontal

# --------- mask drawing helper ----------
def draw_cartoon_mask(img, landmarks, img_w, img_h):
    # Draw a stylized visor over eyes / nose
    # approximate eye centers from some landmark indices
    try:
        left_eye = landmarks[33]  # approx
        right_eye = landmarks[263]
        nose = landmarks[1]
    except Exception:
        return img
    lx, ly = int(left_eye.x * img_w), int(left_eye.y * img_h)
    rx, ry = int(right_eye.x * img_w), int(right_eye.y * img_h)
    nx, ny = int(nose.x * img_w), int(nose.y * img_h)
    # draw thick ellipse connecting eyes
    center = ((lx + rx)//2, (ly + ry)//2)
    axes = (int(abs(rx - lx)*0.9), int(abs(ly - ny)*2.2))
    overlay = img.copy()
    cv2.ellipse(overlay, center, axes, 0, 0, 360, (50, 120, 200), -1)
    # transparent overlay
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    # outline
    cv2.ellipse(img, center, axes, 0, 0, 360, (255,255,255), 2)
    return img

# --------- main loop ----------
def main():
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return

    current_mode = 1
    mask_on = True
    last_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur capture")
            break

        # mirror the frame (like a mirror)
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        landmarks = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

        display = frame.copy()

        # choose effect
        if current_mode == 2:
            display = cartoonize(display)
        elif current_mode == 3:
            display = glitch_effect(display)
        elif current_mode == 4:
            # deformation: enlarge eyes and stretch mouth if smiling
            if landmarks:
                # eyes: left 33 region, right 263 region - build small bbox around
                # use nearby landmarks to estimate region size
                def bbox_from_indices(idx_center, radius=12):
                    pt = landmarks[idx_center]
                    cx = int(pt.x * img_w)
                    cy = int(pt.y * img_h)
                    r = int(img_w * 0.06)  # heuristic
                    x = max(0, cx - r)
                    y = max(0, cy - r)
                    return (x, y, min(img_w - x, 2*r), min(img_h - y, 2*r))

                left_bbox = bbox_from_indices(33)
                right_bbox = bbox_from_indices(263)
                display = enlarge_roi(display, left_bbox, scale=1.45)
                display = enlarge_roi(display, right_bbox, scale=1.45)

                # mouth stretch if smiling
                mar = mouth_aspect_ratio(landmarks, img_w, img_h)
                # mar threshold heuristics: open mouth -> larger; smile slightly increases vertical
                if mar > 0.03:
                    # mouth center roughly landmark 0.. use 0 as nose base for placement adjust
                    # approximate mouth bbox using landmarks 61..291 region
                    xs = [int(p.x * img_w) for p in landmarks]
                    ys = [int(p.y * img_h) for p in landmarks]
                    # mouth area heuristics:
                    mouth_x = min(xs[61], xs[291])
                    mouth_x2 = max(xs[61], xs[291])
                    mouth_y = min(ys[13], ys[14])
                    mouth_y2 = max(ys[13], ys[14])
                    w = max(1, mouth_x2 - mouth_x)
                    h = max(1, mouth_y2 - mouth_y)
                    mouth_bbox = (max(0, mouth_x - w), max(0, mouth_y - h), min(img_w, w*3), min(img_h, h*4))
                    display = enlarge_roi(display, mouth_bbox, scale=1.5)

        # draw mask overlay if requested
        if mask_on and landmarks:
            display = draw_cartoon_mask(display, landmarks, img_w, img_h)

        # draw small FPS and mode info
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = int(1/dt) if dt > 0 else fps

        mode_names = {1: "NORMAL", 2: "CARTOON", 3: "GLITCH", 4: "DEFORMATION"}
        cv2.putText(display, f"Mode: {mode_names.get(current_mode,'?')}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, f"FPS: {fps}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Miroir Interactif", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == ord('1'):
            current_mode = 1
        elif key == ord('2'):
            current_mode = 2
        elif key == ord('3'):
            current_mode = 3
        elif key == ord('4'):
            current_mode = 4
        elif key == ord('m'):
            mask_on = not mask_on
        elif key == ord('s'):
            # save snapshot
            ts = int(time.time())
            fname = os.path.join(SAVE_DIR, f"capture_{ts}.png")
            cv2.imwrite(fname, display)
            print("Saved", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
