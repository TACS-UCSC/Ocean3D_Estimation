
**Title**:
**High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations**

**Dataset:**

The dataset is publicly available on Zenodo: https://zenodo.org/records/19116637. This work uses GLORYS subsurface states, including temperature, salinity, zonal velocity (U), and meridional velocity (V), together with satellite surface observations as conditional inputs. 


**Abstract**:

The ocean interior regulates Earth’s climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware generative framework for reconstructing high-resolution three-dimensional ocean states from sparse surface data. Our approach employs a conditional denoising diffusion probabilistic model (DDPM) trained on sea surface height and temperature observations, without reliance on a background dynamical model. By incorporating continuous depth embeddings, the model learns a unified vertical representation and generalizes to previously unseen depths. Applied to the Gulf of Mexico, the framework accurately reconstructs subsurface temperature, salinity, and velocity fields across multiple depths. Evaluations using statistical metrics, spectral analysis, and heat transport diagnostics demonstrate recovery of both large-scale circulation and multiscale variability. These results establish generative diffusion models as a scalable approach for probabilistic ocean reconstruction in data-limited regimes, with implications for climate monitoring and forecasting.


**Methods**:

Our proposed model is a **depth-aware conditional Denoising Diffusion Probabilistic Model (DDPM)** based on the original DDPM framework introduced by Ho et al. [1]. It incorporates depth identifiers and sparse surface observations as conditioning inputs for 3D ocean-state reconstruction.

To assess its performance, we also implement two deterministic baselines adapted for depth-conditioned reconstruction: a depth-aware UNet and a depth-aware Fourier Neural Operator (FNO). **The code for these models is in the Base Models folder.**

In addition, we investigate two hybrid architectures, **UNET+DDPM** and **FNO+DDPM**, which use deterministic model predictions as priors to guide the diffusion-based reconstruction process.

**The depth-aware conditional DDPM** is designed to reconstruct high-resolution 3D ocean dynamics—T, S, U, and V—from sparse surface observations of SSH and SST. **The code for this model is in the DDPM Models folder.**

The model is trained on nine discrete vertical levels: 25.21 m, 55.76 m, 109.73 m, 155.85 m, 222.48 m, 318.13 m, 453.94 m, 643.57 m, and 1062.44 m. To enable a unified vertical representation, each depth is encoded as a continuous normalized scalar conditioning variable. This allows the model to learn a global vertical representation rather than separate depth-specific mappings


**References**
[1]. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 6840–6851, 2020.






