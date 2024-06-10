import cv2
import numpy as np


# Abre el video
cap = cv2.VideoCapture('carrosss.mp4')
# Crea el objeto del sustractor de fondo
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# Umbral de distancia para fusionar contornos cercanos
merge_distance_threshold = 50
# Duración en frames para mantener un rectángulo después de que deja de ser detectado
rect_persistence_time = 30

# Diccionario para almacenar rectángulos y sus contadores de persistencia
rects_dict = {}

# Inicializa el fgmask previo como None
prev_fgmask = None
# Máscara acumulativa para los píxeles que están desapareciendo
disappearing_mask = np.zeros((480, 640), dtype=np.uint8)


def merge_close_rects(rects, threshold):
    merged = True
    while merged:
        merged = False
        new_rects = []
        while rects:
            rect1 = rects.pop(0)
            x1, y1, x2, y2 = rect1
            rect_merged = False
            for i in range(len(rects)):
                rect2 = rects[i]
                x3, y3, x4, y4 = rect2

                # Calculate the distance between rectangles
                dx = max(x1 - x4, x3 - x2, 0)
                dy = max(y1 - y4, y3 - y2, 0)
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance < threshold:
                    # Merge the rectangles
                    new_rect = (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))
                    new_rects.append(new_rect)
                    rects.pop(i)
                    rect_merged = True
                    merged = True
                    break

            if not rect_merged:
                new_rects.append(rect1)

        rects = new_rects
    return rects


while True:
    # Lee un frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona el frame
    frame = cv2.resize(frame, (640, 480))

    # Aplica el sustractor de fondo
    fgmask = fgbg.apply(frame)

    if prev_fgmask is not None:
        # Encuentra los píxeles que están empezando a desaparecer (eran blanco antes y ya no lo son)
        disappearing_pixels = (prev_fgmask == 255) & (fgmask == 0)

        # Marca los píxeles que están desapareciendo en la máscara acumulativa
        disappearing_mask[disappearing_pixels] = 255

    else:
        # Si no hay fgmask previa, reinicia la máscara acumulativa
        disappearing_mask[:] = 0

    # Convierte fgmask a BGR para poder colorear de rojo
    fgmask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    # Dibuja los píxeles que están desapareciendo en rojo en la máscara de primer plano
    fgmask_bgr[disappearing_mask == 255] = [0, 0, 255]

    # Guarda el fgmask actual como el anterior para el próximo frame
    prev_fgmask = fgmask.copy()

    # Encuentra los contornos en la máscara de primer plano
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)  # Asegúrate de que los contornos sean una lista

    # Lista para almacenar los rectángulos delimitadores
    rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append((x, y, x + w, y + h))

    # Fusiona los rectángulos cercanos
    merged_rects = merge_close_rects(rects, merge_distance_threshold)

    # Actualiza los rectángulos persistentes
    new_rects_dict = {}
    for rect in merged_rects:
        new_rects_dict[rect] = rect_persistence_time

    for rect in list(rects_dict.keys()):
        if rect in new_rects_dict:
            new_rects_dict[rect] = rect_persistence_time  # Reinicia el contador
        else:
            rects_dict[rect] -= 1  # Decrementa el contador
            if rects_dict[rect] > 0:
                new_rects_dict[rect] = rects_dict[rect]  # Mantén el rectángulo

    rects_dict = new_rects_dict

    # Dibuja los rectángulos fusionados y persistentes en el frame
    for rect in rects_dict.keys():
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Muestra la máscara de primer plano y el frame original
    cv2.imshow('fgmask', fgmask_bgr)
    cv2.imshow('frame', frame)

    # Espera por la tecla ESC para salir
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

# Libera el video y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
