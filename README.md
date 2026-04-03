
**Title**:
**High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations**

**Dataset:**

The dataset is publicly available on Zenodo: https://zenodo.org/records/19116637. This work uses GLORYS subsurface states, including temperature, salinity, zonal velocity (U), and meridional velocity (V), together with satellite surface observations as conditional inputs. 


**Abstract**:

The ocean interior regulates Earth’s climate but remains sparsely observed due to limited in situ measurements, while satellite observations are restricted to the surface. We present a depth-aware generative framework for reconstructing high-resolution three-dimensional ocean states from sparse surface data. Our approach employs a conditional denoising diffusion probabilistic model (DDPM) trained on sea surface height and temperature observations, without reliance on a background dynamical model. By incorporating continuous depth embeddings, the model learns a unified vertical representation and generalizes to previously unseen depths. Applied to the Gulf of Mexico, the framework accurately reconstructs subsurface temperature, salinity, and velocity fields across multiple depths. Evaluations using statistical metrics, spectral analysis, and heat transport diagnostics demonstrate recovery of both large-scale circulation and multiscale variability. These results establish generative diffusion models as a scalable approach for probabilistic ocean reconstruction in data-limited regimes, with implications for climate monitoring and forecasting.


