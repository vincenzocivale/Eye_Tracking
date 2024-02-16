import torch
import torch.nn as nn
import torch.optim as optim

# Definizione del modello di rete neurale per il gaze-tracking
class GazeTrackingModel(nn.Module):
    def __init__(self):
        super(GazeTrackingModel, self).__init__()

        # Definizione delle torri ConvNet per gli occhi sinistro e destro
        self.left_eye_tower = self.create_conv_tower()
        self.right_eye_tower = self.create_conv_tower()

        # Definizione dei layer fully connected per i landmark degli occhi
        self.fc_landmarks = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        # Definizione dei layer fully connected per la combinazione delle torri
        self.fc_combine_towers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def create_conv_tower(self):
        # Funzione per creare una torre ConvNet
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

    def forward(self, left_eye, right_eye):
        # Passa le regioni degli occhi attraverso le torri ConvNet
        left_eye_features = self.left_eye_tower(left_eye)
        right_eye_features = self.right_eye_tower(right_eye)

        # Riscalamento delle feature maps per la concatenazione
        left_eye_features = left_eye_features.view(left_eye_features.size(0), -1)
        right_eye_features = right_eye_features.view(right_eye_features.size(0), -1)

        # Passa i landmark degli occhi attraverso i layer fully connected
        landmarks_left = self.fc_landmarks(left_eye_features)
        landmarks_right = self.fc_landmarks(right_eye_features)

        # Combina le feature maps delle torri e passa attraverso i layer fully connected
        combined_towers = torch.cat((left_eye_features, right_eye_features), dim=1)
        gaze_output = self.fc_combine_towers(combined_towers)

        return gaze_output

# Inizializzazione del modello
gaze_model = GazeTrackingModel()

# Definizione di una loss function e un ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(gaze_model.parameters(), lr=0.001)
