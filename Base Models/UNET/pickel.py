import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from model import UNet


torch.set_default_dtype(torch.float32)

torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)

LossFunction= "MSE"  # Loss funtion 
EPOCH=500
DATA="UNET_Obs_depth_log_Sub_layers_new" #Denorm and norm task, the normalization part changed.
MODELNAME=DATA+LossFunction+'UNET90_EP'+str(EPOCH)
print("MODELNAME:"+MODELNAME)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc  = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"
unet_ckpt = "/glade/derecho/scratch/nasefi/Ocean3D/Results/BASE_MODELS/UNET/UNET_Obs_depth_log_Sub_layers_newMSEUNET90_EP500.pth"

class DepthMemmapDataset(Dataset):
    def __init__(self, truth_path, cond_path, max_T=None):
        # Load memmaps
        self.truth = np.load(truth_path, mmap_mode="r")  # (Tt, 4, D, H, W)
        self.cond  = np.load(cond_path,  mmap_mode="r")  # (Tc, 6, D, H, W)

        # --- determine shared T safely ---
        T_truth = self.truth.shape[0]
        T_cond  = self.cond.shape[0]
        self.T = min(T_truth, T_cond)

        if max_T is not None:
            self.T = min(self.T, max_T)

        # --- slice both consistently ---
        self.truth = self.truth[:self.T]
        self.cond  = self.cond[:self.T]

        # spatial / depth info
        _, _, self.D, self.H, self.W = self.truth.shape

        print(f"[Dataset] Using T={self.T}, D={self.D}, H={self.H}, W={self.W}")

    def __len__(self):
        return self.T * self.D

    def __getitem__(self, idx):
        t = idx // self.D
        d = idx % self.D

        surface = self.cond[t, :5, d]           # (5, H, W)
        depth_scalar = self.cond[t, 5, d, 0, 0] # scalar
        depth_map = np.full((1, self.H, self.W),
                            depth_scalar,
                            dtype=np.float32)

        x = np.concatenate([surface, depth_map], axis=0)  # (6, H, W)
        return torch.from_numpy(x).float()



dataset = DepthMemmapDataset(
    truth_loc,
    cond_loc,
    max_T=None)

loader = DataLoader(
    dataset,
    batch_size=16,        # tune for GPU
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

TRAIN_T = 11000
T_total = dataset.T
D = dataset.D
H, W = dataset.H, dataset.W

train_N = TRAIN_T * D
test_N  = (T_total - TRAIN_T) * D

print(f"Train samples: {train_N}")
print(f"Test samples:  {test_N}")




model = UNet(in_channels=6, out_channels=4).to(device)
model.load_state_dict(torch.load(unet_ckpt, map_location=device))
model.eval()


N = len(dataset)  # T * D

save_dir = "/glade/derecho/scratch/nasefi/Ocean3D/Results/BASE_MODELS/condition_unet"
os.makedirs(save_dir, exist_ok=True)

train_tmp = f"{save_dir}/unet_pred_train_flat.tmp"
test_tmp  = f"{save_dir}/unet_pred_test_flat.tmp"

unet_train = np.memmap(
    train_tmp,
    dtype="float32",
    mode="w+",
    shape=(train_N, 4, H, W)
)

unet_test = np.memmap(
    test_tmp,
    dtype="float32",
    mode="w+",
    shape=(test_N, 4, H, W)
)



train_idx = 0
test_idx  = 0
global_idx = 0  # counts (t * D + d)

with torch.no_grad():
    for batch_idx, x in enumerate(loader):
        x = x.to(device)
        y_hat = model(x).cpu().numpy()  # (B,4,H,W)
        bsz = y_hat.shape[0]

        for i in range(bsz):
            t = global_idx // D  # recover timestep

            if t < TRAIN_T:
                unet_train[train_idx] = y_hat[i]
                train_idx += 1
            else:
                unet_test[test_idx] = y_hat[i]
                test_idx += 1

            global_idx += 1

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx} | "
                f"train_written={train_idx} | "
                f"test_written={test_idx}"
            )

# Take everything I’ve written to this memmap so far and physically save it to disk immediately.”
unet_train.flush()
unet_test.flush()

del unet_train, unet_test

os.rename(train_tmp, f"{save_dir}/unet_pred_train_flat.npy")
os.rename(test_tmp,  f"{save_dir}/unet_pred_test_flat.npy")

print("✅ UNet train/test predictions saved successfully")

print("Expected train:", train_N)
print("Expected test :", test_N)
print("Actual train  :", train_idx)
print("Actual test   :", test_idx)









