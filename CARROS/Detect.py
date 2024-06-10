import cv2
import torch
import time
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Diccionario que asocia los números de clase con los nombres de las clases
class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    4: "Ambulance"
}


# Función para procesar un frame de la cámara y mostrar los resultados en tiempo real
def process_video():
    # Capturar video desde la cámara 1
    cap1 = cv2.VideoCapture(0)

    # Capturar video desde la cámara 2
    cap2 = cv2.VideoCapture(1)

    # Variable para mantener el tiempo
    start_time = time.time()

    while True:
        # Capturar frames de ambas cámaras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Verificar que ambos frames se hayan capturado correctamente
        if not ret1 or not ret2:
            break

        # Redimensionar los frames a una resolución más baja
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # Convertir los frames de OpenCV a formato PIL (RGB)
        frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_pil1 = Image.fromarray(frame_rgb1)

        frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame_pil2 = Image.fromarray(frame_rgb2)

        # Realizar la inferencia en los dos frames
        results1 = model(frame_pil1, size=640)  # Aumentar el tamaño de entrada
        results2 = model(frame_pil2, size=640)  # Aumentar el tamaño de entrada

        # Obtener todas las detecciones de vehículos para la cámara 1
        vehicle_detections1 = get_vehicle_detections(results1)

        # Obtener todas las detecciones de vehículos para la cámara 2
        vehicle_detections2 = get_vehicle_detections(results2)

        # Dibujar cuadros delimitadores en los frames para las detecciones de la cámara 1
        draw_detections(frame1, vehicle_detections1)

        # Dibujar cuadros delimitadores en los frames para las detecciones de la cámara 2
        draw_detections(frame2, vehicle_detections2)

        # Contar el número de vehículos detectados en cada cámara
        num_vehicles1 = len(vehicle_detections1)
        num_vehicles2 = len(vehicle_detections2)

        # Mostrar el número de detecciones en cada ventana
        cv2.putText(frame1, f'Vehículos detectados: {num_vehicles1}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(frame2, f'Vehículos detectados: {num_vehicles2}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Mostrar un círculo de color verde o rojo dependiendo del número de detecciones en cada ventana
        if num_vehicles1 > num_vehicles2:
            cv2.circle(frame1, (50, 50), 30, (0, 255, 0), -1)  # Mostrar un círculo verde en la ventana de la cámara 1
            cv2.circle(frame2, (50, 50), 30, (0, 0, 255), -1)  # Mostrar un círculo rojo en la ventana de la cámara 2
        elif num_vehicles1 < num_vehicles2:
            cv2.circle(frame1, (50, 50), 30, (0, 0, 255), -1)  # Mostrar un círculo rojo en la ventana de la cámara 1
            cv2.circle(frame2, (50, 50), 30, (0, 255, 0), -1)  # Mostrar un círculo verde en la ventana de la cámara 2

        # Mostrar el video en dos ventanas separadas
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)

        # Verificar si han pasado 5 segundos desde la última impresión
        if time.time() - start_time >= 5:
            # Imprimir el número total de vehículos detectados en la consola
            print(f'Vehículos camara 1: {num_vehicles1}')
            print(f'Vehículos camara 2: {num_vehicles2}')
            # Reiniciar el tiempo
            start_time = time.time()

        # Esperar 1 milisegundo (se puede ajustar si es necesario)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


# Función para obtener todas las detecciones de vehículos
def get_vehicle_detections(results):
    vehicle_detections = []
    vehicle_class_ids = [2, 3, 5, 7]  # Nuevos IDs de las clases de vehículos
    for class_id in vehicle_class_ids:
        class_detections = results.xyxy[0][results.xyxy[0][:, 5] == class_id]
        vehicle_detections.extend(class_detections)
    return vehicle_detections


# Función para dibujar cuadros delimitadores en el frame para las detecciones
def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        class_name = class_names.get(int(class_id), 'Unknown')
        color = (0, 255, 0) if class_id == 2 or class_id == 3 else (0, 0, 255) if class_id == 4 else (255, 0, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)


# Llamar a la función para procesar el video en tiempo real
process_video()
