import sys
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import torch
import yaml
import logging
from datetime import datetime
from pprint import pformat
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_path = "/glade/derecho/scratch/nasefi/Ocean3D/Ocean3D_Estimation/setup.yaml"
with open(yaml_path, "r") as f:
    setup = yaml.safe_load(f)

config_path = "/glade/derecho/scratch/nasefi/Ocean3D/Ocean3D_Estimation/ddpm_config.yml"
with open(config_path, 'r') as h:
    hyperparam_dict = yaml.load(h, Loader=yaml.FullLoader)

# Extract hyperparameters
timesteps = hyperparam_dict["timesteps"]
beta_start = hyperparam_dict["beta_start"]
beta_end = hyperparam_dict["beta_end"]
power = hyperparam_dict["power"]
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


# Get current time for model naming
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")


# Setup paths and device
output_dir = setup["output_dir"]
models_dir = setup["models_dir"]
logging_dir = setup["logging_dir"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)


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


# Create output directories
path_outputs_model = f"{output_dir}/{model_name}"
if not os.path.exists(path_outputs_model):
    printlog(f"Creating directory: {path_outputs_model}")
    os.makedirs(path_outputs_model)
    
path_outputs_model_timesteps = f"{path_outputs_model}/fno_ddpm_timesteps"
if not os.path.exists(path_outputs_model_timesteps):
    printlog(f"Creating directory: {path_outputs_model_timesteps}")
    os.makedirs(path_outputs_model_timesteps)




data_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_level(1_to_9)[SST, sal, u, v]_5Dimcond_6ch_Obs[SSH,SSH, mask_ssh, land, Log_norm]_pwr(2.0)_law(0.015)_hy(128,256)_1stDec_ep143/DDPM_predictions_Obs_ep143_11000_11100.npy"
unet_ddpm_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_(1_to_9)[SST, sal, u, v]_5Dcond_10ch_Obs[SSH,SST,masks,land, Log_norm_depth_ids,4_UNet]_pwr(2.0)_law(0.015)_hy(128,256)Silu_24thJan/UNET_DDPM_obs_predictions_silu_ep127_11000_11100.npy"
fno_ddpm_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_(1_to_9)[SST, sal, u, v]_5Dcond_10ch_Obs[SSH,SST,masks,land, Log_norm_depth_ids,4_FNO]_pwr(2.0)_law(0.015)_hy(128,256)Silu_5thFeb/FNO_DDPM_obs_predictions_silu_ep66_11000_11100.npy"

save_predictions = np.load(data_path)
unet_ddpm_predictions= np.load(unet_ddpm_path)
fno_ddpm_predictions = np.load(fno_ddpm_path)
print("ddpm shape", save_predictions.shape)
print("unet_ddpm shape", unet_ddpm_predictions.shape)
print("fno_ddpm_predictions shape", fno_ddpm_predictions.shape)


# # # 2. Load data
truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"

land_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/sub_10maskchannels.npy"
land_np = np.load(land_loc)   # (T, 10, H, W)
truth_np = np.load(truth_loc, mmap_mode="r")   # (T, 4, D, H, W)
cond_np  = np.load(cond_loc,  mmap_mode="r")   # (T, 6, D, H, W)


# shapes: (T, 4, D, H, W)
ddpm_predictions      = save_predictions
fno_ddpm_predictions  = fno_ddpm_predictions
unet_ddpm_predictions = unet_ddpm_predictions


# Generate samples for t = 11000 to t = 11100  (100 samples)
start_idx = 11000
end_idx   = 11100     

total_samples = end_idx - start_idx

land_np = land_np[start_idx:end_idx]
test_truth = truth_np[start_idx:end_idx]
test_cond  = cond_np[start_idx:end_idx]


T, Cin, D, H, W = test_truth.shape
print("Important truth/cond.", test_truth.shape, test_cond.shape)

# Load the NumPy array 
land = torch.from_numpy(land_np).float() [:, 1:10, :, :] #size 9
data  = torch.from_numpy(test_truth).float()
cond  = torch.from_numpy(test_cond).float() 

T, Cin, D, H, W = data.shape
data_flat = data.permute(0,2,1,3,4).reshape(T*D, 4, H, W)
cond_flat = cond.permute(0,2,1,3,4).reshape(T*D, 6, H, W)


# PLOT prediction vs truth with LAND MASK applied
titles = ["Temp", "Salinity", "U", "V"]
save_dir = f"{output_dir}/{model_name}/depth"

os.makedirs(save_dir, exist_ok=True)

## RECONSTRUCT ALL DEPTH LEVELS USING TRAINED DEPTH_NORM CHANNEL
save_root = f"{output_dir}/{model_name}"
os.makedirs(save_root, exist_ok=True)


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

############# add cartopy codes here #####################################
depth_array = np.array([
    25.21141, 55.76429, 109.7293, 155.8507,
    222.4752, 318.1274, 453.9377, 643.5668,
    1062.44
])

local_id = 0
t_index = start_idx + local_id

depth_labels = ["222 m", "453.9 m", "643 m"]
depth_ids = [4, 6, 7]
min_max_ids = [5, 7, 8]

# ############# Final FFT ################################
def compute_y_avg_fft(data):
    """Compute FFT along y-axis and average the magnitudes"""
    data_fft = np.fft.rfft(data, axis=1)
    return np.mean(np.abs(data_fft), axis=0)


print("\n===== Starting FFT Analysis Over 100 Samples =====")


models = {
    "DDPM": ddpm_predictions,
    "UNet+DDPM": unet_ddpm_predictions,
    "FNO+DDPM": fno_ddpm_predictions,
    "Truth": test_truth
}


chs = ["Temp", "Salinity", "U", "V"]
ch_idx_map = {"Temp": 0, "Salinity": 1, "U": 2, "V": 3}
fft_data = {
    model: {ch: {d: [] for d in depth_ids} for ch in chs}
    for model in models
}


# # ------------------------- LOOP ------------------------------ 
land_arr = land_np[:, 1:10] 

for i, (d_global, mm_id) in enumerate(zip(depth_ids, min_max_ids)):
    d_local = i   # 0,1,2 corresponding to selected depths
   
    minmax_path = (
        f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/"
        f"DATA_norm_np/min_max_normalize/levels_min_max_normalize/"
        f"level{mm_id}_min_max_all.npy"
    )

    for s in range(total_samples):

        land_mask = land_arr[s, d_global]

        for model_name, model_arr in models.items():

            slice_norm = model_arr[s, :, d_global]  # (4,H,W)
            slice_denorm = denormalize(slice_norm, land_mask, minmax_path)

            for ch_name in chs:
                cidx = ch_idx_map[ch_name]
                fft_data[model_name][ch_name][d_global].append(
                    compute_y_avg_fft(slice_denorm[cidx])[2:-1]
                )

model_styles = {
    "DDPM":      dict(color="green",  lw=1.8),
    "UNet+DDPM": dict(color="blue",   lw=1.8),
    "FNO+DDPM":  dict(color="red",    lw=1.8),
    "Truth":     dict(color="black",  lw=2.2)
}

fig, axes = plt.subplots(
    nrows=len(chs),          # 4 variables
    ncols=len(depth_ids),    # 3 depths
    figsize=(11, 9),
    sharex=True,
    sharey=False
)

for j, d in enumerate(depth_ids):
    for i, ch_name in enumerate(chs):

        ax = axes[i, j]
        ks = None

        for model_name in models:
            arr = np.stack(fft_data[model_name][ch_name][d], axis=0)
            ks = np.arange(arr.shape[1]) + 2

            ax.plot(
                ks,
                arr.mean(axis=0),
                **model_styles[model_name],
                label=model_name
            )
      
            if model_name != "Truth":  # optional shading only for models
                ax.fill_between(
                    ks,
                    arr.mean(axis=0) - arr.std(axis=0),
                    arr.mean(axis=0) + arr.std(axis=0),
                    color=model_styles[model_name]["color"],
                    alpha=0.25
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        leg= ax.legend(
    loc="upper right",          # good default for spectra
    fontsize=7,
    frameon=True,
    framealpha=0.85,
    edgecolor="none",
    handlelength=2
)
        for text in leg.get_texts():
            text.set_fontweight("bold")
        if j == 0:
            ax.set_ylabel(f"{ch_name}\nAmplitude", fontsize=10, fontweight="bold")
    
        if i == 0:
           ax.set_title(depth_labels[j], fontsize=11, fontweight="bold")

        if i == len(chs) - 1:
            ax.set_xlabel("Wavenumber k", fontsize=10,  fontweight="bold")

        ax.tick_params(axis="both", which="both", labelsize=8, width=1.5)

        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(
    f"{path_outputs_model_timesteps}/S4_Spectrum.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


###### Pearson Coefficient Correlation #######################
def get_cc(preds, truths):
    """
    Calculate Pearson correlation coefficient over ocean (ignoring NaNs/land).

    Args:
        preds (np.ndarray): Predicted values with shape (steps, lat, lon, channels)
        truths (np.ndarray): Ground truth values with shape (steps, lat, lon, channels)

    Returns:
        np.ndarray: Average Pearson CC over steps for each channel (length = num_channels)
    """
    assert preds.shape == truths.shape
    steps, _, _, channels = preds.shape
    ccs = np.empty((steps, channels))

    for istep in range(steps):
        for ch in range(channels):
            pred = preds[istep, :, :, ch].flatten()
            truth = truths[istep, :, :, ch].flatten()

            # Remove NaNs (mask land)
            valid_mask = ~np.isnan(pred) & ~np.isnan(truth)
            if np.sum(valid_mask) == 0:
                ccs[istep, ch] = np.nan
            else:
                ccs[istep, ch] = pearsonr(pred[valid_mask], truth[valid_mask])[0]
       
    avg_cc_per_channel = ccs.mean(axis=0)
    std_cc_per_channel = ccs.std(axis=0)
    return avg_cc_per_channel, std_cc_per_channel 

channels = ["Temp", "Salinity", "U", "V"]

# ####################################### Normalized RMSE ############################################
def get_nrmse_std(preds, truths):
    """
    Compute Normalized RMSE using standard deviation normalization.

    NRMSE = RMSE / std(truth)

    Args:
        preds:  np.ndarray of shape (steps, lat, lon, channels)
        truths: np.ndarray of identical shape

    Returns:
        tuple:
            - mean_nrmse: shape (channels,)
            - std_nrmse:  shape (channels,)
    """
    assert preds.shape == truths.shape
    steps, _, _, channels = preds.shape
    
    # Compute RMSE for every sample and channel
    nrmse_all = np.empty((steps, channels))

    for step in range(steps):
        for ch in range(channels):
            pred_ch  = preds[step, :, :, ch]
            truth_ch = truths[step, :, :, ch]

            # RMSE
            diff = pred_ch - truth_ch
            rmse = np.sqrt(np.nanmean(diff**2))

            # Standard deviation of truth (denominator)
            std_truth = np.nanstd(truth_ch)

            # Avoid division by zero
            if std_truth == 0 or np.isnan(std_truth):
                nrmse_all[step, ch] = np.nan
            else:
                nrmse_all[step, ch] = rmse / std_truth

    # Mean and std across steps
    mean_nrmse = np.nanmean(nrmse_all, axis=0)
    std_nrmse  = np.nanstd(nrmse_all, axis=0)

    return mean_nrmse, std_nrmse

nrmse_dict = {}
nrmse_std_dict = {}

######################### SSIM #######################################
def get_ssim(preds, truths):
    assert preds.shape == truths.shape
    steps, H, W, channels = preds.shape
    ssims = np.empty((steps, channels))

    for step in range(steps):
        for ch in range(channels):
            pred = preds[step, :, :, ch]
            truth = truths[step, :, :, ch]

            valid_mask = ~np.isnan(pred) & ~np.isnan(truth)
            if np.sum(valid_mask) == 0:
                ssims[step, ch] = np.nan
                continue

            # Mask land as NaN
            pred_masked = pred.copy()
            truth_masked = truth.copy()
            pred_masked[~valid_mask] = np.nan
            truth_masked[~valid_mask] = np.nan

            data_range = np.nanmax(truth_masked) - np.nanmin(truth_masked)
            if data_range == 0:
                ssims[step, ch] = np.nan
            else:
                ssims[step, ch] = ssim(
                    truth_masked,
                    pred_masked,
                    data_range=data_range,
                    gaussian_weights=True,
                    use_sample_covariance=False
                )

    return np.nanmean(ssims, axis=0), np.nanstd(ssims, axis=0)


metrics = ["NRMSE", "CC", "SSIM"]
models_list = ["DDPM", "UNet+DDPM", "FNO+DDPM"]


values = {depth_labels[i]: {} for i in range(len(depth_ids))}
errors = {depth_labels[i]: {} for i in range(len(depth_ids))}

model_arrays = {
    "DDPM": ddpm_predictions,
    "UNet+DDPM": unet_ddpm_predictions,
    "FNO+DDPM": fno_ddpm_predictions
}

##########################################
# A figure with 3 * 3 panels. each row for each depth and each colurmn for metrics NormRMSE, CC, SSIM. 
#Each panel we have y axis = NormRMSE, x axis = 4 diff channels and each channel we have (UNET+DDPM, DDPM, FNO+DDPM)

channels = ["Temp", "Salinity", "U", "V"]
model_order = ["DDPM", "UNet+DDPM", "FNO+DDPM"]
metric_names = ["NRMSE", "CC", "SSIM"]


results = {
    i: {m: {} for m in model_order}
    for i in range(len(depth_ids))
}

for i, (d_global, mm_id) in enumerate(zip(depth_ids, min_max_ids)):
    d_local = i   # 0,1,2 corresponding to selected depths
   
    minmax_path = (
        f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/"
        f"DATA_norm_np/min_max_normalize/levels_min_max_normalize/"
        f"level{mm_id}_min_max_all.npy"
    )

    land_arr = land_np[:, d_global]  # (samples, H, W)

    for model_name, model_arr in model_arrays.items():

        preds_denorm = []
        truth_denorm = []

        for s in range(total_samples):
            land_mask = land_arr[s]

            pred = denormalize(model_arr[s, :, d_global], land_mask, minmax_path)
            tru  = denormalize(test_truth[s, :, d_global], land_mask, minmax_path)

            preds_denorm.append(np.moveaxis(pred, 0, -1))   # (H,W,C)
            truth_denorm.append(np.moveaxis(tru, 0, -1))

        preds_denorm = np.stack(preds_denorm)   # (S,H,W,C)
        truth_denorm = np.stack(truth_denorm)

        results[i][model_name]["NRMSE"] = get_nrmse_std(preds_denorm, truth_denorm)
        results[i][model_name]["CC"]    = get_cc(preds_denorm, truth_denorm)
        results[i][model_name]["SSIM"]  = get_ssim(preds_denorm, truth_denorm)


colors = {
    "DDPM": "tab:green",
    "UNet+DDPM": "tab:blue",
    "FNO+DDPM": "tab:red"
}


fig, axes = plt.subplots(
    nrows=len(depth_ids),
    ncols=len(metric_names),
    figsize=(12, 9),
    sharey=False
)

bar_width = 0.22
x = np.arange(len(channels))

for i, d_global in enumerate(depth_ids):
    depth_name = depth_labels[i]

    for j, metric in enumerate(metric_names):
        ax = axes[i, j]

        for k, model in enumerate(model_order):
            mean, std = results[i][model][metric]

            ax.bar(
                x + (k - 1) * bar_width,
                mean,
                yerr=std,
                width=bar_width,
                color=colors[model],
                label=model if i == 0 and j == 0 else None,
                capsize=3
            )

        ax.set_xticks(x)
        ax.set_xticklabels(channels, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

        if j == 0:
            ax.set_ylabel(depth_name, fontsize=10, fontweight="bold")

        if i == 0:
            ax.set_title(metric, fontsize=11, fontweight="bold")

        ax.tick_params(axis="y", labelsize=8)

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
fig.legend(
    model_order,
    loc="lower center",
    ncol=3,
    prop={"weight": "bold", "size": 10},
    frameon=False
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(
    f"{path_outputs_model_timesteps}/S6_all_metric.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

#############################################################
