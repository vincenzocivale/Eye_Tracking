import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
from torchvision import transforms



# Funzione che dato l'immagine in ingresso e il modello da utilizzare, restituisce l'immagine dell'occhio destro e sinistro,
# dimensioni 128x128. Con l'immagine dell'occhio sinistro specchiata per facilitare l'addestramento.
def detect_eyes(face_detector, landmark_predictor, image):
    # Converti l'immagine in scala di grigi (Dlib funziona meglio su immagini in scala di grigi)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rileva i volti nell'immagine
    faces = face_detector(gray_image)

    # Imposta le dimensioni desiderate per l'immagine ritagliata dell'occhio
    desired_eye_size = (128, 128)

    # Trova le coordinate dei landmark facciali per ogni volto rilevato
    for face in faces:
        landmarks = landmark_predictor(image, face)
        right_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        left_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calcola le coordinate del rettangolo di ritaglio centrato sull'occhio destro
        x_right, y_right, w_right, h_right = cv2.boundingRect(np.array(right_eye_coords))
        cx_right, cy_right = x_right + w_right // 2, y_right + h_right // 2
        x_crop_right = max(0, cx_right - desired_eye_size[0] // 2)
        y_crop_right = max(0, cy_right - desired_eye_size[1] // 2)

        right_eye_image = image[y_crop_right:y_crop_right + desired_eye_size[1],
                          x_crop_right:x_crop_right + desired_eye_size[0]]

        # Calcola le coordinate del rettangolo di ritaglio centrato sull'occhio sinistro
        x_left, y_left, w_left, h_left = cv2.boundingRect(np.array(left_eye_coords))
        cx_left, cy_left = x_left + w_left // 2, y_left + h_left // 2
        x_crop_left = max(0, cx_left - desired_eye_size[0] // 2)
        y_crop_left = max(0, cy_left - desired_eye_size[1] // 2)

        left_eye_image = image[y_crop_left:y_crop_left + desired_eye_size[1],
                         x_crop_left:x_crop_left + desired_eye_size[0]]

        # Rifletti l'immagine dell'occhio sinistro orizzontalmente per semplificare l'addestramento successivo
        left_eye_image = cv2.flip(left_eye_image, 1)

        return right_eye_image, left_eye_image


def plot_eyes(right_eye_image, left_eye_image):
    # Plotta le porzioni ritagliate degli occhi
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2RGB))
    plt.title('Right Eye')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2RGB))
    plt.title('Left Eye')

    plt.show()


def preprocess_eye_images(right_eye, left_eye):
    # Define a transform to normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply the transform to each eye image
    right_eye = transform(right_eye)
    left_eye = transform(left_eye)

    # Add batch dimension to the images
    right_eye = right_eye.unsqueeze(0)
    left_eye = left_eye.unsqueeze(0)

    return right_eye, left_eye



# Carica il rilevatore del volto di Dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\cical\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

# Carica l'immagine
image = cv2.imread(r"C:\Users\cical\Downloads\msg.jpg")

right_eye, left_eye = detect_eyes(face_detector, landmark_predictor, image)

right_eye, left_eye = preprocess_eye_images(right_eye,left_eye)

