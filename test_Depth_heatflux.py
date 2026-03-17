"""
Calculate ocean heat fluxes and heat content from GLORYS NetCDF files.

Functions:
1. Meridional heat flux = ρ * CP * v * T (then do zonally average)
2. Zonal heat flux = ρ * CP * u * T (then do meridionally average)

where:
    ρ = seawater density (kg/m³)
    CP = specific heat capacity of seawater (J/(kg·K))
    v = northward velocity (m/s)
    u = eastward velocity (m/s)
    T = temperature (°C or K)
"""
import torch
import yaml
import numpy as np
import os
from importlib import reload
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os



truth_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/DATA_alldepths_wo_land_min_max.npy"
cond_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/DATA_Depth_min_max/final_obs_normdepth_6ch_log.npy"

land_loc = "/glade/derecho/scratch/nasefi/Ocean3D/DATA_Sub/sub_10maskchannels.npy"
land_np = np.load(land_loc)   # (T, 10, H, W)
truth_np = np.load(truth_loc, mmap_mode="r")   # (T, 4, D, H, W)
cond_np  = np.load(cond_loc,  mmap_mode="r")   # (T, 6, D, H, W)


data_path = "/glade/derecho/scratch/nasefi/Ocean3D/Results/ddpm_output/DDPM_Depth_min_max_level(1_to_9)[SST, sal, u, v]_5Dimcond_6ch_Obs[SSH,SSH, mask_ssh, land, Log_norm]_pwr(2.0)_law(0.015)_hy(128,256)_1stDec_ep143/DDPM_predictions_Obs_ep143_11000_11100.npy"

ddpm_data = np.load(data_path)
print("data shape", ddpm_data.shape)

# Generate samples for t = 11000 to t = 11100  (100 samples)
start_idx = 11000
end_idx   = 11100     #→ produces 100 samples


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

# Constant parameters
RHO = 1035.# seawater density
CP = 3985.# specific heat capacity
lats = np.linspace(17, 31, 169)
lons = np.linspace(-98.99999, -74.08333, 300)

# florida strait parameters
target_lon = -81.0
lat_min = 23.0
lat_max = 25.25

lon_idx = np.argmin(np.abs(lons - target_lon))
lat_mask = (lats >= lat_min) & (lats <= lat_max)
lat_indices = np.where(lat_mask)[0]


def calculate_layer_weights(depths, depth_limit):
    """
    Calculate layer thicknesses using mid-point boundaries.

    Layer boundaries are defined at mid-points between depth levels.
    For first layer: boundary is at surface (0m)
    For last layer: boundary is extrapolated beyond the last depth level

    Parameters:
    -----------
    depths : array
        Depth levels in meters
    depth_limit : float
        depth limit for vertical average
    Returns:
    --------
    weight : array
        weight of each layer
    """
    n = len(depths)
    UB = np.zeros(n)
    LB = np.zeros(n)
    thick = np.zeros(n)

    # First boundary at surface
    UB[0] = 0.0

    # Interior boundaries at mid-points
    for i in range(1, n):
        UB[i] = (depths[i-1] + depths[i]) / 2.0
        LB[i-1] = UB[i]
    # Last boundary (extrapolate)
    LB[n-1] = depths[-1] + (depths[-1] - depths[-2]) / 2.0
    # Calculate thickness
    for i in range(0, n):
        if LB[i] <= depth_limit:
            thick[i] = LB[i]-UB[i]
        elif UB[i] < depth_limit:
            thick[i] = depth_limit-UB[i]
        else:
            thick[i] = np.nan
            
    weight = thick/depth_limit
    
    return weight

depths = np.array([
    25.21141, 55.76429, 109.7293, 155.8507,
    222.4752, 318.1274, 453.9377, 643.5668,
    1062.44
])


########### calculate meridional heat flux ################
def calculate_meridional_heat_flux(
    ddpm_data,        # (T, 4, D, H, W)       # (T, 4, D, H, W)  (optional, for comparison)
    lats, lons,
    depths,
    target_lat=21.797,
    lon_min=-87.0,
    lon_max=-84.78,
    depth_limit_for_avg=200.0
):
    """
    Calculate zonally averaged meridional heat flux from NumPy arrays.
    """

    T_steps, _, D, H, W = ddpm_data.shape

    # --------------------------------------------------
    # 1. Find transect indices
    # --------------------------------------------------
    lat_idx = np.argmin(np.abs(lats - target_lat))
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lon_indices = np.where(lon_mask)[0]

    print(f"Transect latitude: {lats[lat_idx]:.3f}°N")
    print(f"Longitude range: {lons[lon_indices[0]]:.2f}° to {lons[lon_indices[-1]]:.2f}°")
    print(f"Number of longitude points: {len(lon_indices)}")

    # 2. Extract variables
    T_ddpm = ddpm_data[:, 0]   # temperature
    V_ddpm = ddpm_data[:, 3]   # meridional velocity

    # 3. Vertical weights
    weights = calculate_layer_weights(depths, depth_limit_for_avg)

    # 4. Allocate outputs
    heat_flux_td = np.zeros((T_steps, D))
    depth_avg_flux = np.zeros(T_steps)

    # 5. Compute heat flux
    for t in range(T_steps):
        for d in range(D):
            temp = T_ddpm[t, d, lat_idx, lon_indices]
            vel  = V_ddpm[t, d, lat_idx, lon_indices]

            Q = RHO * CP * vel * temp   # W/m^2

            heat_flux_td[t, d] = np.nanmean(Q)

        depth_avg_flux[t] = np.nansum(heat_flux_td[t] * weights)

    # --------------------------------------------------
    # 6. Package results
    # --------------------------------------------------
    results = {
        "heat_flux": heat_flux_td,          # (time, depth)
        "depth_avg_heat_flux": depth_avg_flux,
        "depth": depths,
        "latitude": lats[lat_idx],
        "longitude_range": (lon_min, lon_max),
        "units": "W/m^2"
    }

    return results


results_meridional_ddpm = calculate_meridional_heat_flux(
    ddpm_data=ddpm_data,
    lats=lats,
    lons=lons,
    depths=depths,
    target_lat=21.797,
    lon_min=-87.0,
    lon_max=-84.78,
    depth_limit_for_avg=200.0
)



def calculate_zonal_heat_flux(
    ddpm_data,        # (T, 4, D, H, W)
    lats, lons,
    depths,
    target_lon=-81.0,
    lat_min=23.0,
    lat_max=25.25,
    depth_limit_for_avg=200.0
):
    """
    Calculate meridionally averaged zonal heat flux (NumPy version).
    """

    T_steps, _, D, H, W = ddpm_data.shape

    # 1. Find transect indices

    lon_idx = np.argmin(np.abs(lons - target_lon))
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lat_indices = np.where(lat_mask)[0]

    print(f"Transect longitude: {lons[lon_idx]:.3f}°")
    print(f"Latitude range: {lats[lat_indices[0]]:.2f}° to {lats[lat_indices[-1]]:.2f}°")
    print(f"Number of latitude points: {len(lat_indices)}")


    # 2. Extract variables
   
    T_ddpm = ddpm_data[:, 0]   # temperature
    U_ddpm = ddpm_data[:, 2]   # zonal velocity (IMPORTANT)

    # --------------------------------------------------
    # 3. Vertical weights
    # --------------------------------------------------
    weights = calculate_layer_weights(depths, depth_limit_for_avg)

    # --------------------------------------------------
    # 4. Allocate outputs
    # --------------------------------------------------
    heat_flux_td = np.zeros((T_steps, D))
    depth_avg_flux = np.zeros(T_steps)

    # --------------------------------------------------
    # 5. Compute heat flux
    # --------------------------------------------------
    for t in range(T_steps):
        for d in range(D):
            temp = T_ddpm[t, d, lat_indices, lon_idx]
            vel  = U_ddpm[t, d, lat_indices, lon_idx]

            Q = RHO * CP * vel * temp   # W/m^2

            heat_flux_td[t, d] = np.nanmean(Q)

        depth_avg_flux[t] = np.nansum(heat_flux_td[t] * weights)

    # --------------------------------------------------
    # 6. Package results
    # --------------------------------------------------
    results = {
        "heat_flux": heat_flux_td,          # (time, depth)
        "depth_avg_heat_flux": depth_avg_flux,
        "depth": depths,
        "longitude": lons[lon_idx],
        "latitude_range": (lat_min, lat_max),
        "units": "W/m^2"
    }

    return results

results_zonal_ddpm = calculate_zonal_heat_flux(
    ddpm_data=ddpm_data,
    lats=lats,
    lons=lons,
    depths=depths,
    depth_limit_for_avg=200.0
)
############################################################
heat_flux = results_meridional_ddpm["heat_flux"]   # (T, D)
depths = results_meridional_ddpm["depth"]

time = np.arange(heat_flux.shape[0]) + start_idx  # actual time indices
save_dir = f"{output_dir}/{model_name}/depth"

os.makedirs(save_dir, exist_ok=True)


for d, depth in enumerate(depths):
    plt.figure(figsize=(6, 3))
    plt.plot(time, heat_flux[:, d], lw=2)

    plt.xlabel("Time index")
    plt.ylabel("Meridional Heat Flux (W/m²)")
    plt.title(f"Meridional Heat Flux at depth = {depth:.1f} m")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/Meridional_Heat_Flux_depth{d+1}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


heat_flux_zonal = results_zonal_ddpm["heat_flux"]   # (T, D)
depths_zonal = results_zonal_ddpm["depth"]

time = np.arange(heat_flux_zonal.shape[0]) + start_idx  # actual time indices


for d, depth in enumerate(depths_zonal):
    plt.figure(figsize=(6, 3))
    plt.plot(time, heat_flux_zonal[:, d], lw=2)

    plt.xlabel("Time index")
    plt.ylabel("Zonal Heat Flux (W/m²)")
    plt.title(f"Zonal Heat Flux at depth = {depth:.1f} m")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/Zonal_Heat_Flux_depth{d+1}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

########## Call function only with truth data.
truth_slice = truth_np[start_idx:end_idx]   # (100, 4, 9, 169, 300)

results_meridional_truth = calculate_meridional_heat_flux(
    ddpm_data=truth_slice,
    lats=lats,
    lons=lons,
    depths=depths,
    target_lat=21.797,
    lon_min=-87.0,
    lon_max=-84.78,
    depth_limit_for_avg=200.0
)

results_zonal_truth = calculate_zonal_heat_flux(
    ddpm_data=truth_slice,
    lats=lats,
    lons=lons,
    depths=depths,
    depth_limit_for_avg=200.0
)

########## plots for both ddpm and truth###########
heat_flux_ddpm  = results_meridional_ddpm["heat_flux"]     # (T, D)
heat_flux_truth = results_meridional_truth["heat_flux"]    # (T, D)
depths = results_meridional_ddpm["depth"]

time = np.arange(heat_flux_ddpm.shape[0]) + start_idx

save_dir = f"{output_dir}/{model_name}/depth"
os.makedirs(save_dir, exist_ok=True)

Qy_ddpm   = results_meridional_ddpm["depth_avg_heat_flux"]
Qy_truth  = results_meridional_truth["depth_avg_heat_flux"]

Qx_ddpm   = results_zonal_ddpm["depth_avg_heat_flux"]
Qx_truth  = results_zonal_truth["depth_avg_heat_flux"]

time = np.arange(len(Qy_ddpm)) + start_idx


#######################################################
heat_flux_ddpm_zonal  = results_zonal_ddpm["heat_flux"]
heat_flux_truth_zonal = results_zonal_truth["heat_flux"]
depths = results_zonal_ddpm["depth"]

time = np.arange(heat_flux_ddpm_zonal.shape[0]) + start_idx

fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

# --- Yucatan Channel (Meridional) ---
axes[0].plot(time, Qy_truth, color="black", lw=2, label="GLORYS")
axes[0].plot(time, Qy_ddpm, color="royalblue", lw=2, label="DDPM")
axes[0].set_ylabel("Heat Flux (W/m²)")
axes[0].set_title("Yucatan Channel (Meridional)")
axes[0].legend(frameon=False)
axes[0].grid(True)

# --- Florida Strait (Zonal) ---
axes[1].plot(time, Qx_truth, color="black", lw=2, label="GLORYS")
axes[1].plot(time, Qx_ddpm, color="royalblue", lw=2, label="DDPM")
axes[1].set_ylabel("Heat Flux (W/m²)")
axes[1].set_xlabel("Time index")
axes[1].set_title("Florida Strait (Zonal)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("Figure5_Proposal_Avg_HeatFlux_Comparison.png", dpi=300)
plt.show()

########### the longitude–depth transect of 𝑣T ###########
def compute_meridional_vT_transect(
    data,              # (T, 4, D, H, W)
    lats, lons, depths,
    time_index,        # single time index (e.g. 11000)
    target_lat=26.0,
    lon_min=-99.0,
    lon_max=-74.0,
    rho=1035.0,
    cp=3985.0
):
    """
    Compute meridional heat-flux transect q_v = rho * cp * v * T

    Returns:
        vT_transect : (lon, depth)
        lon_sel     : (lon,)
        depth       : (depth,)
    """

    # --- find latitude index ---
    lat_idx = np.argmin(np.abs(lats - target_lat))

    # --- longitude indices ---
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lon_idx = np.where(lon_mask)[0]

    # --- extract variables ---
    T = data[time_index, 0, :, lat_idx, lon_idx]   # (D, lon)
    V = data[time_index, 3, :, lat_idx, lon_idx]   # (D, lon)

    # --- compute heat-flux density ---
    vT = rho * cp * V * T                           # (D, lon)

    # transpose to (lon, depth) for plotting
    vT = vT.T

    return vT, lons[lon_idx], depths

####### Example at T=11000
vT_ddpm, lon_sel, depth = compute_meridional_vT_transect(
    data=ddpm_data,
    lats=lats,
    lons=lons,
    depths=depths,
    time_index=0,          # index inside ddpm_data (0–99)
    target_lat=26.0,
    lon_min=-99,
    lon_max=-74
)

truth_slice = truth_np[start_idx:end_idx]  # (100, 4, D, H, W)

vT_truth, lon_sel, depth = compute_meridional_vT_transect(
    data=truth_slice,
    lats=lats,
    lons=lons,
    depths=depths,
    time_index=0,
    target_lat=26.0,
    lon_min=-99,
    lon_max=-74
)


# --------------------------------------------------
# 1. Define color limits from TRUTH only
# --------------------------------------------------
vmin = np.nanmin(vT_truth)
vmax = np.nanmax(vT_truth)

print(f"Colorbar limits (from Truth): [{vmin:.2e}, {vmax:.2e}]")

# --------------------------------------------------
# 2. Create side-by-side plot
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4),
    sharey=True,
    constrained_layout=True
)

# ---- Truth ----
pcm_truth = axes[1].pcolormesh(
    lon_sel, -depth, vT_truth,
    cmap="turbo",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("Truth", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Longitude", fontsize=14, fontweight="bold")
# axes[1].set_ylabel("Depth (m)", fontsize=14, fontweight="bold")
axes[1].set_ylim(-depth.max(), 0)

# ---- DDPM ----
pcm_ddpm = axes[0].pcolormesh(
    lon_sel, -depth, vT_ddpm,
    # cmap="RdBu_r",
    cmap="turbo",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("DDPM", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Longitude", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Depth (m)", fontsize=14, fontweight="bold")
axes[0].set_ylim(-depth.max(), 0)

# --------------------------------------------------
# 3. Shared colorbar
# --------------------------------------------------
cbar = fig.colorbar(
    pcm_truth,
    ax=axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.02
)
cbar.set_label(
    r"$\mathbf{\rho c_p vT}\;(\mathbf{W\,m^{-2}})$",
    fontsize=12
)


cbar.ax.tick_params(labelsize=14)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")

offset = cbar.ax.yaxis.get_offset_text()
offset.set_fontweight("bold")
offset.set_fontsize(14)

for ax in axes:
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')



# 4. Super-title and save

fig.suptitle(
    "Meridional Heat-Flux Transect at 26°N",
    fontsize=16,
    fontweight="bold"
)

plt.savefig(
    "Final11000_Depth_Meridional_HeatFlux_Transect_ddpm.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()




