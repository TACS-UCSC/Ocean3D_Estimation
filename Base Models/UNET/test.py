import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math
from model import UNet
from torch.utils.data import DataLoader, Dataset
import os


torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)

LossFunction= "MSE"  # Loss funtion 
EPOCH=500
DATA="UNET_Obs_depth_log_Sub_layers_new"
model_name=DATA+LossFunction+'UNET_EP'+str(EPOCH)
print("MODELNAME:"+model_name)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Denorm function
def denormalize(pred, land_mask, minmax_file):
    """
    pred:       (4, H, W)  OR (B, 4, H, W)
    land_mask:  (H, W)     values 0 or 1
    minmax_file: .npy file containing shape (4,2) → [min,max] for each variable
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(land_mask, torch.Tensor):
        land_mask = land_mask.detach().cpu().numpy()

    minmax = np.load(minmax_file)   # shape (4,2)
    out = pred.copy()

    if pred.ndim == 3:
        # (4, H, W)
        for v in range(4):
            mn, mx = minmax[v]
            out[v] = out[v] * (mx - mn) + mn
            out[v] *= land_mask
        return out

    elif pred.ndim == 4:
        # (B, 4, H, W)
        B = pred.shape[0]
        for b in range(B):
            for v in range(4):
                mn, mx = minmax[v]
                out[b, v] = out[b, v] * (mx - mn) + mn
                out[b, v] *= land_mask
        return out
    else:
        raise ValueError("denormalize() expected shape (4,H,W) or (B,4,H,W)")




torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset (SAME as training)
# -----------------------------
class DepthMemmapDataset(Dataset):
    def __init__(self, truth_path, cond_path, t_start, t_end):
        self.truth = np.load(truth_path, mmap_mode="r")[t_start:t_end]
        self.cond  = np.load(cond_path,  mmap_mode="r")[t_start:t_end]

        self.T, _, self.D, self.H, self.W = self.truth.shape

    def __len__(self):
        return self.T * self.D

    def __getitem__(self, idx):
        t = idx // self.D
        d = idx % self.D

        y = self.truth[t, :, d]        # (4,H,W)

        surface = self.cond[t, :5, d]  # (5,H,W)
        depth_scalar = self.cond[t, 5, d, 0, 0]
        depth_map = np.full((1, self.H, self.W), depth_scalar, dtype=np.float32)

        x = np.concatenate([surface, depth_map], axis=0)

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy())
        )

# -----------------------------
# Paths
# -----------------------------
truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc  = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"
land_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/sub_10maskchannels.npy"
# model_path = "/glade/derecho/scratch/nasefi/Ocean3D/UNET_Obs_depth_log_Sub_layers_newMSEUNET90_EP500.pth"
model_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/BASE_MODELS/UNET_Obs_depth_log_Sub_layers_newMSEUNET90_EP500.pth"

truth_np = np.load(truth_loc, mmap_mode="r")
cond_np  = np.load(cond_loc,  mmap_mode="r")
land_np  = np.load(land_loc)


# Generate samples for t = 11000 to t = 11100  (100 samples)
start_idx = 11000
end_idx   = 11100     #→ produces 100 samples

total_samples = end_idx - start_idx

land_np = land_np[start_idx:end_idx][:, 1:10, :, :]
test_truth = truth_np[start_idx:end_idx]
test_cond  = cond_np[start_idx:end_idx]

test_ds = DepthMemmapDataset(
    truth_loc,
    cond_loc,
    t_start=11000,
    t_end=11100
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# Load model

model = UNet(in_channels=6, out_channels=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = nn.MSELoss(reduction="sum")

# -----------------------------
# Test loop
# -----------------------------
total_loss = 0.0
num_samples = 0

# Your training depths
depth_array = np.array([
    25.21141, 55.76429, 109.7293, 155.8507,
    222.4752, 318.1274, 453.9377, 643.5668,
    1062.44
])

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        num_samples += labels.numel()

rmse = torch.sqrt(torch.tensor(total_loss / num_samples))
print(f"✅ Test RMSE (normalized): {rmse.item():.6e}")
save_root = "/glade/derecho/scratch/nasefi/Ocean3D/Results/UNET_output"
output_dir= "/glade/derecho/scratch/nasefi/Ocean3D/Results/UNET_output"
t_samples = 5
for local_id in range(t_samples):   # 0..99
    t_index = start_idx + local_id

    print(f"\nGenerating sample {local_id} → t_index {t_index}")

    for d_train in range(len(depth_array)):

        # ---------- build UNET input ----------
        surface = cond_np[t_index, :5, d_train]        # (5,H,W)
        depth_scalar = cond_np[t_index, 5, d_train, 0, 0]
        depth_map = np.full((1, surface.shape[1], surface.shape[2]),
                            depth_scalar, dtype=np.float32)

        x = np.concatenate([surface, depth_map], axis=0)   # (6,H,W)
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)

        # ---------- UNET forward ----------
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()   # (4,H,W)

        truth = truth_np[t_index, :, d_train]  # (4,H,W)
        # land_mask = land_np[t_index, d_train]  # (H,W)
        land_mask = land_np[local_id, d_train]


        # ---------- denormalize ----------
        minmax_path = f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_norm_np/min_max_normalize/levels_min_max_normalize/level{d_train+1}_min_max_all.npy"
        
        # pred_dn  = denormalize(pred,  land_mask, minmax_path)
        # truth_dn = denormalize(truth, land_mask, minmax_path)
        truth_dn = truth.copy()
        pred_dn = pred.copy()

        ocean = land_mask == 1
        pred_dn[:, ~ocean]  = np.nan
        truth_dn[:, ~ocean] = np.nan

        # ---------- plotting ----------
        save_dir = f"{output_dir}/{model_name}/t{t_index}/depth{d_train}"
        os.makedirs(save_dir, exist_ok=True)

        titles = ["T", "Salinity", "U", "V"]
        for i in range(4):
            vmin, vmax = np.nanmin(truth_dn[i]), np.nanmax(truth_dn[i])

            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            im0= axs[0].imshow(pred_dn[i], cmap="coolwarm", vmin=vmin, vmax=vmax)
            axs[0].set_title("UNET")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            im1=axs[1].imshow(truth_dn[i], cmap="coolwarm", vmin=vmin, vmax=vmax)
            axs[1].set_title("Truth")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(f"{titles[i]} | depth {depth_array[d_train]:.2f} m")
            plt.tight_layout()
            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")

            plt.savefig(f"{save_dir}/{titles[i]}.png", dpi=300)
            plt.close()







