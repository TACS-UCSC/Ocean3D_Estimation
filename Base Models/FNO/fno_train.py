import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
# from loss_Spectrum import spectral_sqr_abs2
from Fno2D import *
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import os

LossFunction= "MSE"  # Loss funtion 
EPOCH=500
DATA="FNO_Obs_depth_log_Sub_layers_21thJan" #Denorm and norm task, the normalization part changed.
MODELNAME=DATA+LossFunction+'FNO_EP'+str(EPOCH)
print("MODELNAME:"+MODELNAME)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"


BASE_DIR = "/glade/derecho/scratch/nasefi/Ocean3D/Results/BASE_MODELS/FNO"
CHECKPOINT_PATH = f"{BASE_DIR}/{MODELNAME}_checkpoint.pth"
BEST_MODEL_PATH = f"{BASE_DIR}/{MODELNAME}.pth"

os.makedirs(BASE_DIR, exist_ok=True)




class DepthMemmapDataset(Dataset):
    def __init__(self, truth_path, cond_path, max_T=None):
        self.truth = np.load(truth_path, mmap_mode="r")  # (T,4,D,H,W)
        self.cond  = np.load(cond_path,  mmap_mode="r")  # (T,6,D,H,W)

        if max_T is not None:
            self.truth = self.truth[:max_T]
            self.cond  = self.cond[:max_T]

        self.T, _, self.D, self.H, self.W = self.truth.shape

    def __len__(self):
        return self.T * self.D

    def __getitem__(self, idx):
        t = idx // self.D
        d = idx % self.D

        y = self.truth[t, :, d]        # (4,H,W)

        surface = self.cond[t, :5, d]  # (5,H,W)
        depth_scalar = self.cond[t, 5, d, 0, 0]  # scalar

        # broadcast depth scalar
        depth_map = np.full((1, self.H, self.W), depth_scalar, dtype=np.float32)

        x = np.concatenate([surface, depth_map], axis=0)  # (6,H,W)
        x = torch.from_numpy(x.copy()).permute(1, 2, 0) # (H,W,6)
        y = torch.from_numpy(y.copy()).permute(1, 2, 0) # (H,W,6)


        return (
            x.clone().float(),
            y.clone().float()
        )

#H, W size is # 169 #300
modes = 129
width = 20


dataset = DepthMemmapDataset(truth_loc, cond_loc, max_T=11000)

train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False,
                          num_workers=4, pin_memory=True)


model = FNO2d(
    in_channels=6,
    out_channels=4,
    modes1=modes,
    modes2=modes,
    width=width
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

start_epoch = 0
best_val_loss = float('inf')
epochs_no_improve = 0
losslist = []
val_loss_list = []




print("T used =", dataset.T)
print("Depth levels =", dataset.D)
print("Total samples =", len(dataset))



criterion = nn.MSELoss()
patience = 20


# Training loop with early stopping
print(f"▶ Training will run from epoch {start_epoch} to {EPOCH-1}")
for epoch in range(start_epoch, EPOCH):
    model.train()
    epoch_losses = []
    print(f"▶ Training epoch {epoch+1}")

    for i, (inputs, labels) in enumerate(train_loader):
        if i % 500 == 0:
            print(f"[Epoch {epoch+1}] batch {i}/{len(train_loader)}")

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losslist.append(avg_epoch_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    print(f'Epoch: {epoch + 1}, Training Loss: {avg_epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')
    # 🔐 ALWAYS save checkpoint (walltime-safe)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "losslist": losslist,
        "val_loss_list": val_loss_list,
    }, CHECKPOINT_PATH)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping!')
            break


print("Finished Training or Stopped Early due to Non-Improvement")

loss_data = pd.DataFrame({
    'Epoch': range(1, len(losslist) + 1),
    'Training Loss': losslist,
    'Validation Loss': val_loss_list
})
loss_data.to_csv(MODELNAME+'.csv', index=False)


# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(losslist, label='Training Loss', marker='o')
plt.plot(val_loss_list, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss of '+ MODELNAME)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(MODELNAME+'.png')
plt.show()

