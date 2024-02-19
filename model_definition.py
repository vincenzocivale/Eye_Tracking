import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.svm import SVR

""" 

1) Fase di Pre-Addestramento del Modello Base: In questa fase, il modello base, un'architettura di rete neurale 
   convoluzionale (CNN) multilivello feed-forward, viene pre-addestrato utilizzando un sottoinsieme del dataset 
   pubblicamente disponibile MIT GazeCapture. Le immagini degli occhi sono utilizzate come input, e il modello è 
   addestrato per predire la posizione dello sguardo sullo schermo del telefono.

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

"""Manca da aggiungere la normalizzazione del batch perchè non so dove tra quali livelli inserirlo"""
# Definizione dell'architettura del modello di base
def build_base_model(input_shape):
    # Define ConvNet towers for each eye
    convnet_tower = models.Sequential([
        layers.Conv2D(32, (7, 7), strides=(2, 2), activation='relu', input_shape=input_shape),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
        layers.AveragePooling2D((2, 2)),
    ])

    # Define fully connected layers for landmarks
    landmarks_fc = models.Sequential([
        layers.Dense(128, activation='relu', name='FC1'),
        layers.Dense(16, activation='relu', name='FC2'),
        layers.Dense(16, activation='relu', name='FC3')
    ])

    # Define additional fully connected layers
    combined_fc = models.Sequential([
        layers.Dense(8, activation='relu', name='FC4'),
        layers.Dense(4, activation='relu', name='FC5')
    ])

    # Define regression head (l'ultimo layer)
    regression_head = layers.Dense(2)

    # Input placeholders for eye images and landmarks
    eye_image_right = layers.Input(shape=input_shape)
    eye_image_left = layers.Input(shape=input_shape)
    eye_landmarks = layers.Input(shape=(4, 2))  # Inner and outer eye corner landmarks

    # Process eye images through ConvNet towers
    processed_image_right = convnet_tower(eye_image_right)
    processed_image_left = convnet_tower(eye_image_left)

    # Process eye landmarks through fully connected layers
    processed_landmarks = landmarks_fc(eye_landmarks)

    # Combine features from ConvNet towers and fully connected layers
    combined_features = layers.concatenate([processed_image_right, processed_image_left, processed_landmarks])

    # Combine features with additional fully connected layers
    combined_features = combined_fc(combined_features)

    # Output regression
    output = regression_head(combined_features)

    # Define the model
    model = models.Model(inputs=[eye_image_right, eye_image_left, eye_landmarks], outputs=output)

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=.016,
        decay_steps=8000,
        decay_rate=0.64,
        staircase=True
    )

    # Configurazione dei parametri dell'addestramento e della regolarizzazione
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model

"""
Non ho capito 'Data augmentation: random eye cropping: Enabled (see tf.image.random crop)'
"""
def train_base_model(dataset, batch_size=256, train_steps=30000):

    # Mischia il dataset e definisci la dimensione del batch
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    # Costruisci il modello
    model = build_base_model((128, 128, 3))  # Assicurati di aver definito correttamente la funzione build_base_model

    # Addestra il modello
    model.fit(dataset, epochs=train_steps)

    return model

input_shape=(128, 128, 3)
# Creazione del modello
base_model = build_base_model(input_shape)

def fine_tune_base_model(base_model, calibration_data, batch_size=256, train_steps=30000):

    # Rimuovi il livello di output finale per il fine-tuning
    base_model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    # Prepara il dataset per l'addestramento
    dataset = tf.data.Dataset.from_tensor_slices(calibration_data)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    # Addestra il modello
    base_model.fit(dataset, epochs=train_steps)

    return base_model

# Estrai le features dal penultimo livello del modello fine-tuned
def extract_features(fine_tuned_model, calibration_data):
    penultimate_layer_output = fine_tuned_model.layers[-2].output

    # Crea un nuovo modello che utilizza il penultimo strato come output
    feature_extraction_model = tf.keras.Model(inputs=fine_tuned_model.input, outputs=penultimate_layer_output)

    # Applica il modello di estrazione delle features al dataset di calibrazione
    features = feature_extraction_model.predict(calibration_data)

    return features

# Addestra un modello di regressione SVR per lo sguardo personalizzato
def train_personalized_gaze_estimation_model(features, targets):
    svr_model = SVR(kernel='rbf', C=20.0, gamma=0.06)
    svr_model.fit(features, targets)
    return svr_model

"""
Manca la parte di estrazione dei dati di calibrazione
"""