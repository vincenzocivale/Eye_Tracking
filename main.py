import data_preprocessing as dp
import dlib
import cv2
def main():
    # Carica il rilevatore del volto di Dlib
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(
        r"C:\Users\cical\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

    # Carica l'immagine
    image = cv2.imread(r"C:\Users\cical\Downloads\msg.jpg")

    right_eye, left_eye = dp.detect_eyes(face_detector, landmark_predictor, image)

    right_eye, left_eye = dp.preprocess_eye_images(right_eye, left_eye)