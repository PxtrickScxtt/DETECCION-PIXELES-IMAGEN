import cv2
import dlib

# Cargar el detector de caras de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el predictor de puntos clave faciales de dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Asegúrate de tener este archivo


# Función para detectar y dibujar los puntos clave faciales en una cara
def detect_landmarks(frame, face):
    landmarks = predictor(frame, face)
    landmark_points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return landmark_points


# Función para procesar el video y detectar los rasgos faciales en tiempo real
def process_video():
    # Capturar video desde la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar frame de la cámara
        ret, frame = cap.read()

        # Verificar si el frame se capturó correctamente
        if not ret:
            break

        # Convertir el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en el frame
        faces = detector(gray)

        # Dibujar rectángulos alrededor de las caras detectadas
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Detectar y dibujar los puntos clave faciales en cada cara
            landmark_points = detect_landmarks(frame, face)

        # Mostrar el video
        cv2.imshow('Cámara', frame)

        # Esperar 1 milisegundo (se puede ajustar si es necesario)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


# Llamar a la función para procesar el video en tiempo real
process_video()
