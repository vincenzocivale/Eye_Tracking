import tensorflow as tf
from sklearn.svm import SVR
from data_preprocessing import preprocess_eye_images

""" 

1) Fase di Pre-Addestramento del Modello Base: In questa fase, il modello base, un'architettura di rete neurale 
   convoluzionale (CNN) multilivello feed-forward, viene pre-addestrato utilizzando un sottoinsieme del dataset 
   pubblicamente disponibile MIT GazeCapture. Le immagini degli occhi sono utilizzate come input, e il modello è 
   addestrato per predire la posizione dello sguardo sullo schermo del telefono. Durante questo processo, vengono 
   utilizzate tecniche di data augmentation, come il ritaglio casuale dell'occhio e l'alterazione casuale della 
   posizione dei landmark degli occhi, per rendere il modello più robusto a variazioni nelle condizioni di illuminazione, 
   movimento della fotocamera e rumore.

2) Fase di Estrazione delle Feature: Dopo il pre-addestramento, viene estratta una rappresentazione ad alto livello 
   dal modello base ottimizzato. In particolare, l'output dell'ultimo strato del modello base viene utilizzato come 
   rappresentazione delle feature per ciascuna immagine di calibrazione.

3) Fase di Addestramento del Modello Personalizzato: Nell'ultima fase, viene addestrato un modello di regressione 
   vettoriale di supporto (SVR) per ogni partecipante e blocco di tempo. Le feature estratte dalle immagini di calibrazione 
   vengono utilizzate come input, mentre i dati reali dello sguardo vengono utilizzati come output per addestrare 
   il modello SVR. Durante l'addestramento, viene utilizzata una tecnica di "leave-one-task-out setup", in cui i dati 
   di calibrazione dell'inseguimento liscio vengono utilizzati per l'addestramento e i dati di calibrazione dei punti 
   vengono utilizzati per la valutazione. Infine, il modello personalizzato addestrato per ciascun partecipante viene 
   utilizzato per stimare lo sguardo durante l'attività di calibrazione del punto.
   
4) Fase di inferenza, in cui il modello di base pre-addestrato e il modello di regressione vettoriale di supporto (SVR) 
   personalizzato vengono applicati in sequenza a un'immagine per generare la stima finale dello sguardo personalizzata
   
"""

# Definizione dell'architettura del modello di base
def create_model(input_shape):
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (7, 7), strides=(2, 2), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Flatten layer to transition from convolutional layers to fully connected layers
        tf.keras.layers.Flatten(),
        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        # Last fully connected layer without activation
        tf.keras.layers.Dense(2)
    ])

    # Configurazione dei parametri dell'addestramento e della regolarizzazione
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.016, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


# Creazione del modello
base_model = create_model()

# Compilazione del modello base
base_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])


# Estrai le feature dall'ultimo layer del modello base
base_model_features = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Addestramento del modello personalizzato per ogni partecipante
def train_personalized_model(calibration_data_features, ground_truth_gaze):
    personalized_model = SVR(kernel='rbf', C=20.0, gamma=0.06)
    personalized_model.fit(calibration_data_features, ground_truth_gaze)
    return personalized_model

# Funzione per estrarre le feature di calibrazione
def extract_calibration_features(calibration_frames):
    # Implementa la logica per estrarre le feature di calibrazione
    # Utilizza base_model_features per ottenere le feature dal modello base
    calibration_features = base_model_features.predict(calibration_frames)
    return calibration_features


calibration_data_frames = [...]  # Frame di calibrazione
ground_truth_gaze = [...]  # Dati reali dello sguardo per l'addestramento del modello personalizzato

# Estrai le feature di calibrazione
calibration_data_features = extract_calibration_features(calibration_data_frames)

# Addestramento del modello personalizzato SVR
personalized_model = train_personalized_model(calibration_data_features, ground_truth_gaze)


# Funzione per l'inferenza dello sguardo personalizzato
def infer_custom_gaze(image_path=r"C:\Users\cical\Downloads\msg.jpg", land_mark_predictor_path=r"C:\Users\cical\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"):
    # 1. Pre-elaborazione dell'immagine (assicurati di pre-elaborare l'immagine come durante l'addestramento)
    preprocessed_image = preprocess_eye_images(image_path, land_mark_predictor_path)

    # 2. Estrai le feature dall'immagine utilizzando il modello di base pre-addestrato
    base_model_features = base_model.predict(preprocessed_image)

    # 3. Applica il modello di regressione SVR personalizzato per ottenere la stima dello sguardo personalizzata
    personalized_gaze = personalized_model.predict(base_model_features)

    return personalized_gaze



estimated_gaze = infer_custom_gaze()  # Stima dello sguardo personalizzata
print("Estimated gaze:", estimated_gaze)

