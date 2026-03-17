from importlib import reload
import logging
import yaml
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pformat
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

# Load Setup
yaml_path = "/glade/derecho/scratch/nasefi/Ocean3D/Ocean3D_Estimation/setup.yaml"
with open(yaml_path, "r") as f:
    setup = yaml.safe_load(f)


# Setup paths and device
output_dir = setup["output_dir"]
models_dir = setup["models_dir"]
logging_dir = setup["logging_dir"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

# Import custom modules
import models
reload(models)
from models import simple_unet_new
reload(simple_unet_new)
from models.simple_unet_new  import SimpleUnetCond
from models import loss_functions
reload(loss_functions)
from models.loss_functions import LOSS_FUNCTIONS
import utilities
reload(utilities)
from utilities import n2c, c2n, pthstr, linear_beta_scheduler, cosine_beta_scheduler, power_beta_scheduler


# Get current time for model naming
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")


# Load hyperparameters
config_path = "/glade/derecho/scratch/nasefi/Ocean3D/Ocean3D_Estimation/ddpm_config.yml"
with open(config_path, 'r') as h:
    hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)


# Extract hyperparameters
timesteps = hyperparam_dict["timesteps"]
beta_start = hyperparam_dict["beta_start"]
power = hyperparam_dict["power"]
beta_end = hyperparam_dict["beta_end"]
batch_size = hyperparam_dict["batch_size"]
epochs = hyperparam_dict["epochs"]
loss_function = hyperparam_dict["loss_function"]
loss_function_start = hyperparam_dict["loss_function_start"]
loss_function_start_batch = hyperparam_dict["loss_function_start_batch"]
loss_args_start = hyperparam_dict["loss_args_start"]
loss_args_end = hyperparam_dict["loss_args_end"]
beta_scheduler = hyperparam_dict["beta_scheduler"]
ddpm_arch = hyperparam_dict["ddpm_arch"]
ddpm_params = hyperparam_dict["ddpm_params"]
train_type = hyperparam_dict["train_type"]
lr = hyperparam_dict["lr"]
model_name = hyperparam_dict["model_name"]

# Generate model name if not provided
if model_name is None:
    model_name = f"New_correct_ddpm_unet99s_arch-{ddpm_arch}_time-{current_time}_timesteps-{timesteps}_epochs-{epochs}"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(f"{logging_dir}/ddpm_qgm_losses_{current_time}.log"),
    logging.StreamHandler()
])
printlog = logging.info
printlog("-"*40)
printlog(f"Running ddpm_turb2d.py for {model_name}...")
printlog(f"loaded ddpm_turb2d_config: {pformat(hyperparam_dict)}")
printlog("-"*40)

# Create model directory
model_dir = f"{models_dir}/{model_name}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data dimensions
Nx = 169
Ny = 300
lead = 0  # same time predictions

#data dimensions are (T, C, D, H, W) = (timesteps, channels, depth_levels, height, width) = (T, 4, 9, 169, 300)
# Condition data dimensions are (T, 6, D, 169, 300)
#Load training data
truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"


class DepthMemmapDataset(Dataset):
    def __init__(self, truth_path, cond_path, max_T=None, to_float32=True):

        truth_all = np.load(truth_path, mmap_mode='r')  # (T, 4, D, H, W)
        cond_all  = np.load(cond_path,  mmap_mode='r')  # (T, 6, D, H, W)

        T_total = truth_all.shape[0]

        # Optional truncation
        if max_T is not None:
            T_use = min(max_T, T_total)
            self.truth_mm = truth_all[:T_use]
            self.cond_mm  = cond_all[:T_use]
        else:
            self.truth_mm = truth_all
            self.cond_mm  = cond_all

        self.T = self.truth_mm.shape[0]
        self.C = self.truth_mm.shape[1]   # 4
        self.D = self.truth_mm.shape[2]   # 9 depth levels
        self.H = self.truth_mm.shape[3]
        self.W = self.truth_mm.shape[4]

        self.to_float32 = to_float32

        # Total samples = T * D (flatten depth dimension)
        self.total_samples = self.T * self.D

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Convert flat index → (t, d)
        t = idx // self.D
        d = idx %  self.D

        # y: (4, H, W)
        y = self.truth_mm[t, :, d, :, :]

        # cond: (6, H, W)  — already includes depth_norm
        cond = self.cond_mm[t, :, d, :, :]

        if self.to_float32:
            y    = y.astype(np.float32, copy=False)
            cond = cond.astype(np.float32, copy=False)

        return torch.from_numpy(y), torch.from_numpy(cond)



dataset = DepthMemmapDataset(truth_loc, cond_loc, max_T=11000)

y0, c0 = dataset[0]
print("y0:", y0.shape)
print("c0:", c0.shape)

# --- Define train/val split BEFORE creating loaders ---
val_size   = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])



print("T used =", dataset.T)
print("Depth levels =", dataset.D)
print("Total samples =", len(dataset))


train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader   = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, pin_memory=True, num_workers=16)


# Create ONE model
model = SimpleUnetCond(**ddpm_params).to(device)


# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)


# --- Now test model forward pass ---
xb_test, cb_test = next(iter(train_loader))
t = torch.randint(0, timesteps, (xb_test.shape[0],), device=device).long()
_ = model(xb_test.to(device), cb_test.to(device), t)



has_nan = has_inf = False
for xb_val, cb_val in val_loader:
    has_nan |= torch.isnan(xb_val).any().item() or torch.isnan(cb_val).any().item()
    has_inf |= torch.isinf(xb_val).any().item() or torch.isinf(cb_val).any().item()

print(f"val: nan={has_nan}, inf={has_inf}")



def ddpm_batch_loss_cond(model, x, c, timesteps, alphas_cumprod,
                        loss_fn_start, loss_fn_end,
                        loss_args_start, loss_args_end,
                        ibatch, loss_function_start_batch):
    B = x.shape[0]
    t = torch.randint(0, timesteps, (B,), device=x.device)
    # noise = torch.randn_like(x)
    noise = torch.randn(x.shape[0], 4, x.shape[2], x.shape[3], device=x.device)
    a_bar_t = alphas_cumprod[t].view(-1,1,1,1)
    noisy = torch.sqrt(a_bar_t) * x + torch.sqrt(1 - a_bar_t) * noise

    pred = model(noisy, c, t)  # <<< USE CONDITION
    # print("Check pred and noise shapes", pred.shape, noise.shape)
    if ibatch <= loss_function_start_batch or loss_function_start_batch == -1:
        return LOSS_FUNCTIONS[loss_fn_start](pred.permute(0,2,3,1), noise.permute(0,2,3,1), **loss_args_start)
    else:
        return LOSS_FUNCTIONS[loss_fn_end](pred.permute(0,2,3,1), noise.permute(0,2,3,1), **loss_args_end)

##############
## TRAINING ##
##############
if train_type != "noise":
    raise ValueError(f"Unsupported train_type: {train_type}. Only 'noise' is supported.")

if True:
    printlog(f"Training {model_name}...")
    # -------------------------------
    # RESUME TRAINING FROM CHECKPOINT
    # -------------------------------
    resume_path = f"{model_dir}/{model_name}_all_epochs.pth"

    start_epoch = 0  # default

    if os.path.exists(resume_path):
        printlog(f"🔄 Found checkpoint: {resume_path} — attempting to resume...")

        ckpt = torch.load(resume_path, map_location="cpu")

        # 1. Epoch resume
        if "epochs_run" in ckpt:
            start_epoch = ckpt["epochs_run"]
            printlog(f"➡️ Resuming from epoch {start_epoch}")

        # 2. Load last model state
        if "models" in ckpt and len(ckpt["models"]) > 0:
            # get last epoch key
            last_key = sorted(
                ckpt["models"].keys(),
                key=lambda x: int(x.split("-")[1])
            )[-1]

            printlog(f"➡️ Loading model weights from {last_key}")
            model.load_state_dict(ckpt["models"][last_key])

    else:
        printlog("🔵 No checkpoint found — starting from epoch 0.")

    printlog(f"🚀 Training will BEGIN at epoch {start_epoch + 1}")

    # Initialize noise scheduler
    if beta_scheduler == "linear":
        betas, alphas, alphas_cumprod = linear_beta_scheduler(beta_start, beta_end, timesteps, device=device)
    elif beta_scheduler == "cosine":
        betas, alphas, alphas_cumprod = cosine_beta_scheduler(timesteps, device=device)
    elif beta_scheduler =="power_law":
        betas, alphas, alphas_cumprod = power_beta_scheduler(timesteps, beta_start, beta_end,  power,  device= device)

    
    # Plot alphas_cumprod for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(timesteps), c2n(alphas_cumprod), label='alphas_cumprod')
    plt.xlabel('Timesteps')
    plt.ylabel('Alphas Cumulative Product')
    plt.title(f'Alphas Cumulative Product over Timesteps\nbeta_start: {beta_start}, beta_end: {beta_end}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{model_dir}/alphas_cumprod.png", dpi=200)
    plt.close()

    loss_batch = []
    ibatch = 0
    
    best_loss = float("inf")
    best_epoch = None
    epochs_since_improvement = 0
    patience = 20  # Stop if no improvement after 10 epochs

    # Training loop
    val_loss_epoch = []   # (epoch_idx, val_loss)
    train_loss_epoch = [] # (epoch_idx, train_loss)


    for epoch in range(start_epoch, epochs):
        printbatch = 0
        model.train()
        epoch_train_sum, epoch_train_count = 0.0, 0
        
        for xb, cb in train_loader:                 # <<< note two tensors
            xb, cb = xb.to(device, non_blocking=True), cb.to(device, non_blocking=True)

            batch_size_actual = xb.shape[0]
            if ibatch <= loss_function_start_batch or loss_function_start_batch == -1:
                loss_use = loss_function_start
            else:
                loss_use = loss_function

            optimizer.zero_grad()
            loss = ddpm_batch_loss_cond(model, xb, cb, timesteps, alphas_cumprod,
                                loss_function_start, loss_function,
                                loss_args_start, loss_args_end,
                                ibatch, loss_function_start_batch)
    
            loss.backward()
            optimizer.step()
            
            # bookkeeping (unchanged)
            loss_batch.append([ibatch, loss.item()])
            epoch_train_sum += loss.item()
            epoch_train_count += 1

            if ibatch >= printbatch:
                printlog(f"Epoch [{epoch+1}/{epochs}], ibatch {ibatch+1}, loss_use: {loss_use}, Loss: {loss.item():.8f}")
                printbatch = ibatch + 10
            ibatch += 1

        ######### val loss ###############
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb_val, cb_val in val_loader:
                # xbv = xb_val.to(device)
                # cbv = cb_val.to(device)
                xbv = xb_val.to(device, non_blocking=True)
                cbv = cb_val.to(device, non_blocking=True)
                batch_size_actual = xbv.shape[0]
                # evaluate with the 'end' loss (stable) or mirror the same branch using ibatch
                vloss = ddpm_batch_loss_cond(model, xbv, cbv, timesteps, alphas_cumprod,
                                        loss_function_start, loss_function,
                                        loss_args_start, loss_args_end,
                                        ibatch, loss_function_start_batch)
                val_sum += vloss.item()
                val_count += 1

        epoch_train_loss = epoch_train_sum / max(epoch_train_count, 1)
        epoch_val_loss   = val_sum         / max(val_count, 1)
        train_loss_epoch.append([ibatch, epoch_train_loss])
        val_loss_epoch.append([ibatch, epoch_val_loss])
        printlog(f"[Epoch {epoch+1}] train_batches={epoch_train_count} val_batches={val_count}")
        printlog(f"[Epoch {epoch+1}] train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f}")

        # Track epoch loss
        current_epoch_loss = epoch_val_loss
        if current_epoch_loss < best_loss - 1e-6:

            # remove old best model if it exists
            if best_epoch is not None:
                old_path = f"{model_dir}/{model_name}_{best_epoch}_best.pth"
                if os.path.exists(old_path):
                    os.remove(old_path)

            # update best model info
            best_loss = current_epoch_loss
            best_epoch = epoch + 1

            save_path = f"{model_dir}/{model_name}_{best_epoch}_best.pth"
            torch.save(model.state_dict(), save_path)

            epochs_since_improvement = 0
            printlog(f"✨ New best VAL loss: {best_loss:.8f} at epoch {best_epoch}")

        else:
            epochs_since_improvement += 1
            printlog(f"No VAL improvement for {epochs_since_improvement} epoch(s).")

        
        # Plot training progress
        loss_batch_arr = np.array(loss_batch) if len(loss_batch) else np.zeros((0,2))
        train_epoch_arr = np.array(train_loss_epoch) if len(train_loss_epoch) else np.zeros((0,2))
        val_epoch_arr   = np.array(val_loss_epoch)   if len(val_loss_epoch)   else np.zeros((0,2))

        plt.figure()
        if loss_batch_arr.size:
            plt.plot(loss_batch_arr[:,0], loss_batch_arr[:,1], label="train (batch)", alpha=0.4)
        if train_epoch_arr.size:
            plt.plot(train_epoch_arr[:,0], train_epoch_arr[:,1], "-o", label="train (epoch)")
        if val_epoch_arr.size:
            plt.plot(val_epoch_arr[:,0], val_epoch_arr[:,1], "-o", label="val (epoch)")
        
        plt.xlabel("Cumulative batch index")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f"{model_dir}/loss_train_vs_val.png", dpi=200)
        plt.close()

    
        checkpoint_path = f"{model_dir}/{model_name}_all_epochs.pth"

        def safe_load_checkpoint(path):
            try:
                if (not os.path.exists(path)) or os.path.isdir(path) or os.path.getsize(path) < 100:
                    return {"epochs_run": 0, "models": {}}
                return torch.load(path, map_location="cpu")  # map to cpu is safest
            except (RuntimeError, EOFError) as e:
                printlog(f"⚠️ Checkpoint at {path} is corrupt ({e}). Renaming and starting fresh.")
                try:
                    os.rename(path, path + ".corrupt")
                except Exception as _:
                    pass
                return {"epochs_run": 0, "models": {}}

        checkpoint = safe_load_checkpoint(checkpoint_path)
        # Save model for the current epoch in the dictionary
        checkpoint["models"][f"epoch-{epoch+1}"] = model.state_dict()
        checkpoint["epochs_run"] = epoch + 1

        # Save the entire checkpoint dictionary
        torch.save(checkpoint, checkpoint_path)

        # (Optional) Still save the last epoch separately if needed
        torch.save(model.state_dict(), f"{model_dir}/{model_name}_last_epoch.pth")

        # Update the config file
        hyperparam_dict["epochs_run"] = epoch + 1
        with open(f"{model_dir}/config.yml", 'w') as h:
            yaml.dump(hyperparam_dict, h, default_flow_style=False)
        

        if epochs_since_improvement >= patience:
            printlog(f"Early stopping at epoch {epoch+1}")
            break
        

