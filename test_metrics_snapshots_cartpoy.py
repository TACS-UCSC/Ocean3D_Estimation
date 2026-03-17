import sys
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import torch
import numpy as np
import torch
import yaml
import logging
from datetime import datetime
from pprint import pformat
from models.simple_unet_new import SimpleUnetCond
from scipy.stats import pearsonr
import cartopy.crs as ccrs
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


data_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_level(1_to_9)[SST, sal, u, v]_5Dimcond_6ch_Obs[SSH,SSH, mask_ssh, land, Log_norm]_pwr(2.0)_law(0.015)_hy(128,256)_1stDec_ep143/DDPM_predictions_Obs_ep143_11000_11100.npy"


save_predictions = np.load(data_path)
print("ddpm shape", save_predictions.shape)


###Load data
truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"

land_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/sub_10maskchannels.npy"
land_np = np.load(land_loc)  # (T, 10, H, W)
truth_np = np.load(truth_loc, mmap_mode="r")   # (T, 4, D, H, W)



# Generate samples for t = 11000 to t = 11100  (100 samples)
start_idx = 11000
end_idx   = 11100    

total_samples = end_idx - start_idx

land_np = land_np[start_idx:end_idx]
test_truth = truth_np[start_idx:end_idx]


T, Cin, D, H, W = test_truth.shape
print("Important truth/cond.", test_truth.shape)

# Load the NumPy array 
land = torch.from_numpy(land_np).float() [:, 1:10, :, :] #size 9
data = torch.from_numpy(test_truth.copy()).float()
# cond  = torch.from_numpy(test_cond).float() 

T, Cin, D, H, W = data.shape
data_flat = data.permute(0,2,1,3,4).reshape(T*D, 4, H, W)
# cond_flat = cond.permute(0,2,1,3,4).reshape(T*D, 6, H, W)


# PLOT prediction vs truth with LAND MASK applied
titles = ["Temp", "Salinity", "U", "V"]
save_dir = f"{output_dir}/{model_name}/depth"

os.makedirs(save_dir, exist_ok=True)

# # RECONSTRUCT ALL DEPTH LEVELS USING TRAINED DEPTH_NORM CHANNEL
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

depth_ids = [0, 2, 3]
depth_labels = {0: "25 m", 2: "109 m", 3: "155 m"}

# depth_ids = [4, 6, 7]
# depth_labels = {4: "222 m", 6: "453.9 m", 7: "643 m"}

# min max, for 25 m is level1_min_max_normalize, for 109 m is level3, and for 155 is level 4.
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

fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(
    nrows=2 * len(depth_ids),
    ncols=4,
    hspace=0.22,
    wspace=0.15
)

# Variable indices: 0=T, 1=S, 2=U, 3=V
colorbar_ranges = {
    0: {0: (20, 30), 2: (18, 28), 3: (18, 28)},   # Temperature
    1: {0: (34, 37), 2: (36, 37.5), 3: (36, 37.5)},   # Salinity
    2: {0: (-1, 1), 2: (-1, 1), 3: (-1, 1)},      # U
    3: {0: (-1, 1), 2: (-1, 1), 3: (-1, 1)},      # V
}


for i, d_train in enumerate(depth_ids):
    print(i, d_train)
    row_ddpm  = 2 * i
    row_truth = 2 * i + 1

    land_mask = land[local_id, d_train].cpu().numpy()
    ocean = land_mask == 1

    minmax_path = f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_norm_np/min_max_normalize/levels_min_max_normalize/level{d_train+1}_min_max_all.npy"
    pred_denorm  = denormalize(save_predictions[local_id, :, d_train], land_mask, minmax_path)
    truth_denorm = denormalize(test_truth[local_id, :, d_train], land_mask, minmax_path)

    pred_denorm[:, ~ocean]  = np.nan
    truth_denorm[:, ~ocean] = np.nan

    for j in range(4):

        vmin = np.nanmin(truth_denorm[j])
        vmax = np.nanmax(truth_denorm[j])
        if j in colorbar_ranges and d_train in colorbar_ranges[j]:
            vmin, vmax = colorbar_ranges[j][d_train]

        # ================= DDPM =================
        ax_ddpm = fig.add_subplot(gs[row_ddpm, j], projection=ccrs.PlateCarree())
        im1 = ax_ddpm.pcolormesh(
            lon2d, lat2d, pred_denorm[j],
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            shading="auto", transform=ccrs.PlateCarree()
        )
        ax_ddpm.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax_ddpm.coastlines("10m", linewidth=0.6)

        # ================= TRUTH =================
        ax_truth = fig.add_subplot(gs[row_truth, j], projection=ccrs.PlateCarree())
        im2 = ax_truth.pcolormesh(
            lon2d, lat2d, truth_denorm[j],
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            shading="auto", transform=ccrs.PlateCarree()
        )
        ax_truth.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax_truth.coastlines("10m", linewidth=0.6)

        # ---------- GRIDLINES (ONLY LEFT + BOTTOM) ----------
        for ax in [ax_ddpm, ax_truth]:
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.3,
                color="gray",
                alpha=0.5,
                linestyle="--"
            )

            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = False
            gl.bottom_labels = False

         # Bottom longitude labels ONLY on last depth AND Truth row
            if (i == len(depth_ids) - 1) and (ax is ax_truth):
                gl.bottom_labels = True

            if j == 0:
                gl.left_labels = True

            gl.xlabel_style = {"size": 16, "weight": "bold"}
            gl.ylabel_style = {"size": 16, "weight": "bold"}

        # ---------- DEPTH LABEL ----------
        fig.canvas.draw()
        bbox_ddpm  = ax_ddpm.get_position()
        bbox_truth = ax_truth.get_position()
        y_mid = 0.5 * (bbox_ddpm.y1 + bbox_truth.y0)

        if j == 0:
            fig.text(
                -0.08,
                y_mid,
                f"Depth = {depth_labels[d_train]}",
                fontsize=18,
                fontweight="bold",
                va="center"
            )


        if i == 0:
    # Get column bbox directly from GridSpec (row 0, column j)
            col_bbox = gs[0, j].get_position(fig)

            x_center = 0.5 * (col_bbox.x0 + col_bbox.x1)

            fig.text(
                x_center,
                0.965,          # vertical position (adjust if needed)
                titles[j],
                ha="center",
                va="bottom",
                fontsize=22,
                fontweight="bold"
            )


        # ---------- ROW TITLES ----------
        ax_ddpm.set_title("DDPM", fontsize=16, fontweight="bold", pad=10)
        ax_truth.set_title("Truth", fontsize=16, fontweight="bold", pad=10)

        # ---------- COLORBARS ----------
        for ax, im in zip([ax_ddpm, ax_truth], [im1, im2]):
            cax = inset_axes(
                ax, width="3%", height="75%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=12)
            for t in cb.ax.get_yticklabels():
                t.set_fontweight("bold")

plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.05)

save_dir = os.path.join(save_root, "paper_figures")
os.makedirs(save_dir, exist_ok=True)

plt.savefig(
    os.path.join(save_dir, f"S1_inorder{t_index}.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()




# ############# FFT ################################
def compute_y_avg_fft(data):
    """Compute FFT along y-axis and average the magnitudes"""
    data_fft = np.fft.rfft(data, axis=1)
    return np.mean(np.abs(data_fft), axis=0)

# # FFT WITH YOUR FUNCTION (mean ± std over 100 samples)

print("\n===== Starting FFT Analysis Over 100 Samples =====")

pred_norm = save_predictions              # (100, 4, 9, H, W)
truth_norm = test_truth # normalized truth: (100, 4, 9, H, W)
land_arr   = land_np                      # (100, 9, H, W)

chs = ["Temp", "Salinity", "U", "V"]
ch_idx_map = {"Temp": 0, "Salinity": 1, "U": 2, "V": 3}

# # ------------------------- LOOP ------------------------------ 

fft_ddpm  = {ch: {d: [] for d in depth_ids} for ch in chs}
fft_truth = {ch: {d: [] for d in depth_ids} for ch in chs}
for d in depth_ids:

    minmax_path = f"/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_norm_np/min_max_normalize/levels_min_max_normalize/level{d+1}_min_max_all.npy"
    for s in range(total_samples):

        land_mask = land_arr[s, d]

        pred_slice_norm  = pred_norm[s, :, d]
        truth_slice_norm = truth_norm[s, :, d]

        pred_denorm  = denormalize(pred_slice_norm,  land_mask, minmax_path)
        truth_denorm = denormalize(truth_slice_norm, land_mask, minmax_path)

        for ch_name in chs:
            cidx = ch_idx_map[ch_name]

            fft_ddpm[ch_name][d].append(
                compute_y_avg_fft(pred_denorm[cidx])[2:-1]
            )
            fft_truth[ch_name][d].append(
                compute_y_avg_fft(truth_denorm[cidx])[2:-1]
            )

fig, axes = plt.subplots(
    nrows=len(chs),          # 4 variables
    ncols=len(depth_ids),    # 3 depths
    figsize=(11, 9),
    sharex=True,
    sharey=False
)

for j, d in enumerate(depth_ids):  # columns = depth
    depth_m = depth_array[d]

    for i, ch_name in enumerate(chs):  # rows = variable
        ax = axes[i, j]

        ddpm_arr  = np.stack(fft_ddpm[ch_name][d], axis=0)
        truth_arr = np.stack(fft_truth[ch_name][d], axis=0)

        ks = np.arange(ddpm_arr.shape[1]) + 2

        # ---- DDPM ----
        ax.plot(
            ks,
            ddpm_arr.mean(axis=0),
            color="green",
            lw=1.8
        )
        ax.fill_between(
            ks,
            ddpm_arr.mean(axis=0) - ddpm_arr.std(axis=0),
            ddpm_arr.mean(axis=0) + ddpm_arr.std(axis=0),
            color="green",
            alpha=0.3
        )

        # ---- Truth ----
        ax.plot(
            ks,
            truth_arr.mean(axis=0),
            color="black",
            lw=1.8
        )
        ax.fill_between(
            ks,
            truth_arr.mean(axis=0) - truth_arr.std(axis=0),
            truth_arr.mean(axis=0) + truth_arr.std(axis=0),
            color="black",
            alpha=0.2
        )
        leg = ax.legend(
            handles=[
                plt.Line2D([0], [0], color="green", lw=2, label="DDPM"),
                plt.Line2D([0], [0], color="black", lw=2, label="Truth"),
            ],
            loc="upper right",
            ncol=1,
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            edgecolor="none"
        )

        # Make legend text bold
        for text in leg.get_texts():
            text.set_fontweight("bold")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        # Tick styling
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            width=1.5,
            length=4
        )

        ax.tick_params(
            axis="both",
            which="minor",
            width=1.0,
            length=2
        )

        # Make tick labels bold (robust for log-scale)
        for label in ax.get_xticklabels(which='both'):
            label.set_fontweight('bold')

        for label in ax.get_yticklabels(which='both'):
            label.set_fontweight('bold')

        # ---- Row labels (variables) ----
        if j == 0:
            ax.set_ylabel(f"{ch_name}\nAmplitude", fontsize=10, fontweight="bold")

        # ---- Column titles (depths) ----
        if i == 0:
            # ax.set_title(f"{depth_m:.0f} m", fontsize=11, fontweight="bold")
            ax.set_title(depth_labels[d], fontsize=11, fontweight="bold")


        # ---- X label only bottom row ----
        if i == len(chs) - 1:
            ax.set_xlabel("Wavenumber k", fontsize=10, fontweight="bold")
        
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    f"{path_outputs_model_timesteps}/S2_Spctrum_DDPM_depth.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


