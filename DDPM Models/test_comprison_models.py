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
from models.simple_unet_new import SimpleUnetCond
from scipy.stats import pearsonr
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec


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


# Initialize the model architecture based on ddpm_arch
model = SimpleUnetCond(**ddpm_params).to(device)


ddpm_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_level(1_to_9)[SST, sal, u, v]_5Dimcond_6ch_Obs[SSH,SSH, mask_ssh, land, Log_norm]_pwr(2.0)_law(0.015)_hy(128,256)_1stDec_ep143/DDPM_predictions_Obs_ep143_11000_11100.npy"
ddpm_predictions = np.load(ddpm_path)
print("ddpm shape", ddpm_predictions.shape)


#Load data
truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"

unet_ddpm_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_(1_to_9)[SST, sal, u, v]_5Dcond_10ch_Obs[SSH,SST,masks,land, Log_norm_depth_ids,4_UNet]_pwr(2.0)_law(0.015)_hy(128,256)Silu_24thJan/UNET_DDPM_obs_predictions_silu_ep127_11000_11100.npy"
fno_ddpm_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_(1_to_9)[SST, sal, u, v]_5Dcond_10ch_Obs[SSH,SST,masks,land, Log_norm_depth_ids,4_FNO]_pwr(2.0)_law(0.015)_hy(128,256)Silu_5thFeb/FNO_DDPM_obs_predictions_silu_ep66_11000_11100.npy"

unet_ddpm_predictions= np.load(unet_ddpm_path)
fno_ddpm_predictions = np.load(fno_ddpm_path)

land_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/sub_10maskchannels.npy"
land_np = np.load(land_loc)  # (T, 10, H, W)
truth_np = np.load(truth_loc, mmap_mode="r")   # (T, 4, D, H, W)


# Generate samples for t = 11000 to t = 11100  (100 samples)
start_idx = 11000
end_idx   = 11100     #→ produces 100 samples

total_samples = end_idx - start_idx

land_np = land_np[start_idx:end_idx]
test_truth = truth_np[start_idx:end_idx]

T, Cin, D, H, W = test_truth.shape
print("Important truth/cond.", test_truth.shape)

# Load the NumPy array 
land = torch.from_numpy(land_np).float() [:, 1:10, :, :] #size 9
print("land shape", land.shape)
data = torch.from_numpy(test_truth.copy()).float()


T, Cin, D, H, W = data.shape
data_flat = data.permute(0,2,1,3,4).reshape(T*D, 4, H, W)


# PLOT prediction vs truth with LAND MASK applied
titles = ["Temp", "Salinity", "U", "V"]
save_dir = f"{output_dir}/{model_name}/depth"

os.makedirs(save_dir, exist_ok=True)

##RECONSTRUCT ALL DEPTH LEVELS USING TRAINED DEPTH_NORM CHANNEL
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


depth_ids = [8]        # only 55 m
depth_labels = {8: "1062 m"}
model_names = ["DDPM", "FNO+DDPM", "UNET+DDPM", "Truth"]
titles = [
    r"Temperature [$^\circ$C]",
    r"Salinity [psu]",
    r"U [m/s]",
    r"V [m/s]"
]

lat_min, lat_max = 17, 31
lon_min, lon_max = -98.99999, -74.08333
ny, nx = 169, 300

lats = np.linspace(lat_min, lat_max, ny)
lons = np.linspace(lon_min, lon_max, nx)
lon2d, lat2d = np.meshgrid(lons, lats)

fig = plt.figure(figsize=(26, 20))
gs = gridspec.GridSpec(
    nrows=4,
    ncols=4,
    hspace=0.03,
    wspace=0.18
)
# ---- COLUMN TITLES USING fig.text ----
x_positions = [0.18, 0.40, 0.62, 0.84]

for j, title in enumerate(titles):
    fig.text(
        x_positions[j],
        0.94,
        title,
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold"
    )

# Variable indices: 0=T, 1=S, 2=U, 3=V
colorbar_ranges = {
    0: {8: (4.0, 6.0)},
    1: {8: (34.9, 35.1)},
    2: {5: (-0.5, 0.5), 8: (-0.3, 0.3)},
    3: {5: (-0.5, 0.5), 8: (-0.3, 0.3)},
}


for i, d_train in enumerate(depth_ids):
    print(i, d_train)

    land_mask = land[local_id, d_train].cpu().numpy()
    ocean = land_mask == 1
    print("truth normalized range",
        np.nanmin(test_truth[local_id,:,d_train]),
        np.nanmax(test_truth[local_id,:,d_train]))

    print("ddpm normalized range",
        np.nanmin(ddpm_predictions[local_id,:,d_train]),
        np.nanmax(ddpm_predictions[local_id,:,d_train]))
    

    minmax_path = f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_norm_np/min_max_normalize/levels_min_max_normalize/level{d_train+1}_min_max_all.npy"
    ddpm_denorm  = denormalize(ddpm_predictions[local_id,:,d_train], land_mask, minmax_path)
    fno_ddpm_denorm = denormalize(fno_ddpm_predictions[local_id,:,d_train], land_mask, minmax_path)
    unet_ddpm_denorm = denormalize(unet_ddpm_predictions[local_id,:,d_train], land_mask, minmax_path)
    truth_denorm = denormalize(test_truth[local_id,:,d_train], land_mask, minmax_path)

    ddpm_denorm[:, ~ocean]  = np.nan
    fno_ddpm_denorm[:, ~ocean] = np.nan
    unet_ddpm_denorm[:, ~ocean] = np.nan
    truth_denorm[:, ~ocean] = np.nan

    preds = [
        ddpm_denorm,
        fno_ddpm_denorm,
        unet_ddpm_denorm,
        truth_denorm
    ]

    model_names = ["DDPM", "FNO+DDPM", "UNET+DDPM", "Truth"]
    for row in range(4):  # models
        pred = preds[row]

        for col in range(4):  # variables
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            ax.set_title(model_names[row], fontsize=16, fontweight="bold", pad=10)
            field = pred[col].copy()

            if col in colorbar_ranges and d_train in colorbar_ranges[col]:
                vmin, vmax = colorbar_ranges[col][d_train]
            else:
                vmin = np.nanmin(truth_denorm[col])
                vmax = np.nanmax(truth_denorm[col])

            im = ax.pcolormesh(
                lon2d, lat2d, field,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=ccrs.PlateCarree()
            )

            ax.set_extent([lon_min, lon_max, lat_min, lat_max])
            ax.coastlines("10m", linewidth=0.6)
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.3,
                color="gray",
                alpha=0.5,
                linestyle="--"
            )

            # turn everything off first
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            gl.bottom_labels = False

            # latitude labels only on LEFT column
            if col == 0:
                gl.left_labels = True

            # longitude labels only on BOTTOM row
            if row == 3:
                gl.bottom_labels = True
            # Colorbar
            gl.xlabel_style = {"size": 16, "weight": "bold"}
            gl.ylabel_style = {"size": 16, "weight": "bold"}

            cax = inset_axes(
                ax,
                width="3%",
                height="75%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )

            cb = fig.colorbar(im, cax=cax)

            if col in [2,3]:   # U and V
                cb.set_ticks([-0.3, -0.15, 0, 0.15, 0.3])
            cb.ax.tick_params(labelsize=12)

            for t in cb.ax.get_yticklabels():
                t.set_fontweight("bold")

plt.suptitle(
    f"Supp_Depth = {depth_labels[d_train]}",
    fontsize=24,
    fontweight="bold"
)


plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05)

save_dir = os.path.join(save_root, "paper_figures")
os.makedirs(save_dir, exist_ok=True)

plt.savefig(
    os.path.join(save_dir, f"S9_inorder{t_index}.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()






