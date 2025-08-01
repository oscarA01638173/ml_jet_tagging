##ESTE ENTRENAMIENTO ES COMO DICE EL ARTÍCULO DE PARTICLENET DE LOS CHINOS SIMÉTRICOS. está listo para entrenar.
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F


from PN_original import ParticleNet

import numpy as np

def compute_raw_features_with_points(df):
    data = []
    points = []
    labels = []

    for idx, row in df.iterrows():
        # Extraer los 4-momentos de las 200 partículas
        E  = np.array([row[f'E_{i}']  for i in range(200)])
        PX = np.array([row[f'PX_{i}'] for i in range(200)])
        PY = np.array([row[f'PY_{i}'] for i in range(200)])
        PZ = np.array([row[f'PZ_{i}'] for i in range(200)])

        # Magnitud del momento para calcular η
        p = np.sqrt(PX**2 + PY**2 + PZ**2)

        # ==========================
        # Calcular features por partícula
        # ==========================
        jet_points = []
        jet_features = []

        # Momento total del jet (para Δη, Δφ)
        px_jet, py_jet, pz_jet, e_jet = PX.sum(), PY.sum(), PZ.sum(), E.sum()
        p_jet = np.sqrt(px_jet**2 + py_jet**2 + pz_jet**2)
        eta_jet = 0.5 * np.log((p_jet + pz_jet) / (p_jet - pz_jet))
        phi_jet = np.arctan2(py_jet, px_jet)

        for i in range(200):
            e_i, px_i, py_i, pz_i = E[i], PX[i], PY[i], PZ[i]

            # Padding si la partícula es nula
            if e_i == 0 and px_i == 0 and py_i == 0 and pz_i == 0:
                jet_features.append([0, 0, 0, 0])
                jet_points.append([0, 0])
                continue

            p_i = np.sqrt(px_i**2 + py_i**2 + pz_i**2)
            eta = 0.5 * np.log((p_i + pz_i) / (p_i - pz_i))
            phi = np.arctan2(py_i, px_i)

            delta_eta = eta - eta_jet
            delta_phi = (phi - phi_jet + np.pi) % (2 * np.pi) - np.pi  # en [-π, π]

            jet_features.append([e_i, px_i, py_i, pz_i])
            jet_points.append([delta_eta, delta_phi])

        data.append(np.array(jet_features))     # (200, 4)
        points.append(np.array(jet_points))     # (200, 2)
        labels.append(int(row['is_signal_new']))

    return np.array(points), np.array(data), np.array(labels)



SEED = 442
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 1e-2
INPUT_FEATURES = 4  # 4-momento
NUM_CLASSES = 2


# Cargar los datos por chunks, y seleccionar 1 de cada 20 jets
chunk_size = 100
chunks = []
#datos = pd.read_hdf("train.h5", key="table")
for start in range(0, 1200, chunk_size):
    stop = start + chunk_size
    df_chunk = pd.read_hdf("train.h5", key="table", start=start, stop=stop)

    # Simula event_no con índice global
    global_indices = list(range(start, stop))
    mask = [i % 20 == 0 for i in global_indices]

    df_selected = df_chunk[mask]
    chunks.append(df_selected)

df = pd.concat(chunks, ignore_index=True)
#df=datos
# ===usa función actualizada que retorna points, feats, labels ===
points_list, feats_list, labels_list = compute_raw_features_with_points(df)

print(f"Número de jets: {len(labels_list)}")
print(f"Tamaño de ejemplo de feats: {feats_list[0].shape}")   # (N_particles, 4)
print(f"Tamaño de ejemplo de points: {points_list[0].shape}") # (N_particles, 2)

# points: (N, 100, 2)
# data (features): (N, 100, 4)
# labels: (N,)
'''
points, data, labels = compute_raw_features_with_points(df)

# Ver las features de los primeros 10 jets
for i in range(1):
    print(f"\n🔹 Jet {i} — Label: {labels[i]}")
    print("Features (E, PX, PY, PZ):")
    print(data[i])  # shape: (100, 4)
'''

# =========================================
# Dataset como lista de tuplas por jet
# =========================================
points_list = [p.T.astype(np.float32) for p in points_list]    # (2, 200), float32
feats_list  = [f.T.astype(np.float32) for f in feats_list]     # (4, 200), float32

print(f"Tamaño de ejemplo de feats: {feats_list[0].shape}")   # (N_particles, 4)
print(f"Tamaño de ejemplo de points: {points_list[0].shape}") # (N_particles, 2)

dataset = list(zip(points_list, feats_list, labels_list))


g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g
)


# =====================
# Inicializar modelo
# =====================
model = ParticleNet(
    input_dims=INPUT_FEATURES,
    num_classes=NUM_CLASSES,
    for_inference=False
).to(DEVICE)

# =====================
# Optimizador Lookahead + RAdam
# =====================
import torch_optimizer as topt

base_optimizer = topt.RAdam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.95, 0.999),
    eps=1e-5,
    weight_decay=0.0
)

optimizer = topt.Lookahead(
    base_optimizer,
    k=6,        # número de pasos lookahead
    alpha=0.5   # coef. de actualización
)

from torch.optim.lr_scheduler import LambdaLR

# Parámetros
decay_start = int(NUM_EPOCHS * 0.7)
final_lr_ratio = 0.01  # 1% del valor inicial

# Definimos la lambda
def lr_lambda(current_epoch):
    if current_epoch < decay_start:
        return 1.0
    else:
        progress = (current_epoch - decay_start) / (NUM_EPOCHS - decay_start)
        return final_lr_ratio ** progress  # decaimiento exponencial

# Scheduler
scheduler = LambdaLR(optimizer, lr_lambda)

# =====================
# Entrenamiento
# =====================
    
######################################33    
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Para AUC
    all_labels = []
    all_scores = []

    # Barra de progreso por batch
    loop = tqdm(train_loader, desc=f"🔁 Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

    for points, feats, labels in loop:
        points = points.to(DEVICE)
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE).long()

        # Validación defensiva (opcional)
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print("Features contienen NaN o Inf")
        if torch.isnan(points).any() or torch.isinf(points).any():
            print("Points contienen NaN o Inf")

        optimizer.zero_grad()
        outputs = model(points, feats)  # (B, 2)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Para AUC: calcular probabilidades
        probs = F.softmax(outputs, dim=1)  # (B, 2)
        all_scores.extend(probs[:, 1].detach().cpu().numpy())  # probabilidad clase 1
        all_labels.extend(labels.detach().cpu().numpy())

        # Actualiza la barra de progreso
        loop.set_postfix({
            "loss": loss.item(),
            "acc": 100. * correct / total,
            "lr": optimizer.param_groups[0]["lr"]
        })

    # Scheduler (al final de la epoch)
    scheduler.step()

    # Accuracy final
    acc = correct / total

    # AUC final
    try:
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    except ValueError:
        auc = float('nan')  # por si no hay variedad en las clases

    print(f"📘 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss/total:.4f} - Accuracy: {acc:.4f} - AUC: {auc:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
    

# =====================
# Guardar modelo
# =====================
torch.save(model.state_dict(), "particlenet_model_full.pt")
print("Modelo guardado como particlenet_model_full.pt")
