# Task 1
# [Synthetic Nodule Generation]()

## Source Code Path-
* `fake_nodule_3d/src/fake_nodule_3d/nodule.py` - class Nodule() and class CTScan()
* `fake_nodule_3d/src/fake_nodule_3d/helper_functions.py` - helper functions
* `fake_nodule_3d/src/fake_nodule_3d//config.py` - tunable parameters stored
* `fake_nodule_3d/src/fake_nodule_3d/example.ipynb` - sample usage
* `fake_nodule_3d/src/fake_nodule_3d/simulate_flow_2d.ipynb` - slicewise approach for flow (not used in current approach, just added for reference)

## References used:

### Nodule Shape
<u>*Sieren JC, Smith AR, Thiesse J, Namati E, Hoffman EA, Kline JN, McLennan G. Exploration of the volumetric
composition of human lung cancer nodules in correlated histopathology and computed tomography.*</u> https://pubmed.ncbi.nlm.nih.gov/21371772/

### Nodule Texture
<u>*Li X, Samei E, Delong DM, Jones RP, Gaca AM, Hollingsworth CL, Maxfield CM, Carrico CW, Frush
DP. Three-dimensional simulation of lung nodules for paediatric multidetector array CT*</u> https://pubmed.ncbi.nlm.nih.gov/19153182/

### Spacial location of nodule in CT
<u>*Miki, S., Nomura, Y., Hayashi, N., Hanaoka, S., Maeda, E., Yoshikawa, T., … Abe, O. (2020). Prospective Study of Spatial Distribution of Missed Lung Nodules by Readers in CT Lung Screening Using Computer-assisted Detection*</u> https://sci-hub.se/10.1016/j.acra.2020.03.015

### Nodule types information
<u>*Iwano, S., Nakamura, T., Kamioka, Y., & Ishigaki, T. (2005). Computer-aided diagnosis: A
shape classification of pulmonary nodules imaged by high-resolution CT*</u> https://sci-hub.se/10.1016/j.compmedimag.2005.04.009 


## Methods

## Approach
* for every sub-tissue in a nodule-
    * start with a base shape - sphere / ellipsoid / superellipsoid
    * insert holes to match euler number
    * apply flow to match the compactness value (2d/3d)
    * apply intensity distribution to the tissue

* merge these tissues to generate nodule
* load base chest CT array
* choose location for nodule to be merged (clinically defined areas)
* merge nodule with chest CT scan

## Base shape
* `round`- SPHERICAL
* `ovular`- ELLIPSOIDAL
* `lobulated`- SPHERICAL
* `tentacular`- SUPERELIPSOIDAL
* `spiculated`- SPHERICAL

## Euler Number matching
In 3d structures euler number is defined as `2-2g` ; where g is number of holes present in the structure. we simply take target euler number (specified rnage for every tissue in the paper) and get number of holes (h) that should be present. we randomly choose h locations as centers in the initial sphere. In this code we need to make sure that these holes don't overlap (one of shortcomings of flow matching is that more irregular is the initial structure, worse the deformations get). 

once we have the spere mask with holes (will call this euler-mask in future), we extract boundaries from this mask. func `EXTRACT_ALL_SURFACES` extacts cartesian coordinates of all the points which lie on the boundary of euler-mask (this also includes internal hole boundaries). func `SPLIT_BOUNDARY_INTO_HOLES` splits the entire boundary array into multiple arrays (1st element corrosponds to outer boudnary and others are individual hole boudnaries). we do this because we are about to apply flow to match compactness value. flow performs way better in a closed boundary, hence we apply flow individually for outer boundary each hole's boundary.

## How flow is working?
Params: 
    `sigma_ctrl`: controls the range of deformation for a boudnary point
    `sigma_weight`: controls the amount of deformation for each boundary point
    `num_steps`: numbe of steps we are taking to complete our deformation

we take a boundary and then we choose 8x8x8 grid of control points (8x8 in case of 2d flow) which will influence the boundary. we choose random weights (sclaed by `SIGMA_WEIGHT`) of size (num_steps x K x 3) where k is number of control points (grid points). we compute a gausian kernel using control points and boundary points coordinates (tells us influences of each control point on each boundary point). we use gaussian kernel and weights to compute velocity. this scalar velocity is further used to deform the boundary. (we repeat this `NUM_STEP` times). 

In our main approach, we have used flow to influence entire 3d euler-mask but we have also developed an approach where we break mask into 2d slices and applied flow on these slices. below is the method described-

## 2D Flow approach
one issue with 2d flow is that we just can't simply apply flow on every slice since every time we are choosing random weights. so its not possible to maintain continuity along slices. so we needed some way where we are not changing weights. 

while applying flow we can access boundary at every step and continuity exists in consecutive boundaries (step_n and step_n+1). Let's say `NUM_STEPS`= n, so to maintain continuity- we start with step_n boudnary at slice 0, step_n-1 at slice 1.... so on till step_0 boundary at central slice. we do the same for lower half of the mask- ending at step_n boundary at lowermost slice.

## How flow is related to compactness
we simulated flow for all possible combinations of flow_params in the following ranges-
sigma_ctrl- [4.0, 40.0] (at interval of 2.0)
sigma_weight- [5.0, 100.0] (at interval of 5.0)
num_steps- [10.0, 100.0] (at interval of 10.0)
for each combination we simulated flow 100 times and computed compactness vlaue for the final boundary and got columns for mean and standard deviation of compactness values across 100 iterations. (`3d_flow_data.csv`)

compactness value= `4.pi.(cross sectional area)/(perimeter**2)` (for 2d structures)
compactness value= `36.pi.(volume**2) / (surface_area **3)` (for 3d structures)

once this dataset got created, given a target compactness value, we need to get the best flow parameters. firstly we filter set of params which gives average compactness within some tolerance (kept 0.03 by default). in these sets, we filter further to get set of params which have compactness standard deviation less than a given value (this values is taken from the 1st paper for each tissue type). now from this subset, we choose the set of params with least number of steps to optimize on time complexity.

Exception- we choose set of PARAMS with highest {`SIGMA_WEIGHT`-`SIGMA_CTRL`} value for `SPICULATED` NODULE. also we avoid smooth blending for spiculated nodules to preserve spikes at nodule boundary

## Intensity value distribution
intensity distribution- `c(r) = C * (1 - (r / R)^2)^n` where C is peak intensity value
these contrast values are applied after the entire tissue shape is generated and we apply this distribution slice by slice. now for each point, we have r (distance from center), R (distance of last point in the mask along the vector which connects center and chosen point). 
`n = (6/5)*((n1) - (n1/6)) * (f - 1) + n1`
where n1 and n1/6 are exponents corresponding to conditions in which all or one-sixth of the nodule diameter is contained by a CT slice, respectively. according to the 2nd reference, n1 = 2.0 and n1/6 = 2.4 were used to calculate n in our subsequent simulations (experimentaly found best values). f = R / sphere_radius

## Merging all tissues to form a nodule
We compute a hollowness map for a base tissue (Cancer solid in this case) using gaussian dist. We choose 5 MOST HOLLOW DISTINCT target points in the base tissue. These target points act as centers for remaining tissues to merge. We shrink remaining tissues based on clinically found volumetric composition stats. Finally, we blend all tissues on chosen target points with the base tissue.

## Spacial locations for nodules in a chest CT
lung mask of CT scan was used to get 12 clinically defined regions along with their probablities where nodule can be present in a CT. please note that this paper is biased towards nodules which were missed by radiologists.
The `get_location` function identifies anatomically representative random coordinates within a 3D lung mask from a CT scan (`ct_array` and `lung_mask`). It segments the lung volume into upper, mid, and lower zones along the z-axis and further divides the left and right lungs based on the x-axis. Within each of 12 predefined lung regions (e.g., right\_apical, left\_mid\_dorsal etc), it selects a voxel labeled as lung tissue (mask value 1), using tailored criteria to ensure the voxel lies in a regionally appropriate part (e.g., ventral, dorsal, hilar, lateral) of the lung cross-section. It then returns either (1) a dictionary of region-wise centers, (2) a specific region’s center if requested, or (3) a randomly selected center based on predefined anatomical probabilities if `region="stats"` is specified.

## Merge nodule with chest CT scan 
Inserts a synthetic nodule into a CT volume at a given location using alpha blending.
we normalize the intensity values of nodule to HU range of -1000 to 500 followed by shrinking the nodule to required size. now, we use lung mask and get CT location for nodule to be placed. Alpha blend the CT patch with nodule. Finally, we smoothen the region by applying a gaussian filter.


# Navigation

## 📄 `config.py`

This file contains configuration values or constants (details not extracted due to minimal or missing function definitions).

---



## 📄 `nodule.py`

This is the main file containing the `Nodule` and `CTScan` classes and related logic.

### Class: `Nodule`

#### Methods

| Method | Args | Description |
|--------|------|-------------|
| `__init__` | `self` | Constructor. |
| `modify_params` | `self, tissue_type` | Adjust generation parameters for a given tissue type. |
| `apply_flow_3d` | `self, boundary, flow, weights` | Deform 3D boundary using Gaussian-smooth flow. |
| `smooth_boundary` | `self, boundary, window, polyorder` | Smooth a surface using Savitzky–Golay filter. |
| `merge_tissues_at_hollow` | `self, tissue1, tissue2, insert_center` | Merge tissue2 into hollow region of tissue1. |
| `contrast_profile` | `self, vol_data, C, B, n1, n2, sphere_r` | Assign intensity to nodule using falloff model. |
| `add_random_noise` | `self, vol_data_arr, var` | Add random noise to volume. |
| `add_perlin_noise_3d` | `self, volume, scale, amplitude, seed` | Add Perlin noise for realism. |
| `merge_lobulated_tissues` | `self, tissue_mat` | Merge all 6 tissues using hollowness-guided centers. |
| `tissue_pipeline` | `self, tissue_type` | Full pipeline to synthesize a given tissue volume. |
| `nodule_pipeline` | `self, nodule_type` | Entry function for nodule generation. |
| `visualize_specific_tissue_contribution` | `self, tissue` | Debug visualization for one tissue type. |

---

### Class: `CTScan`

#### Methods

| Method | Args | Description |
|--------|------|-------------|
| `__init__` | `self` | Constructor. |
| `load_ct` | `self, sid` | Load a CT scan from its SeriesInstanceUID. |
| `merge_nodule_with_ct` | `self, ct_mat, nodule_mat, location, alpha, size` | Insert a nodule into a CT at a given location. |
| `smooth_blend` | `self, ct_patch, nodule_patch, nodule_mask, alpha, sigma` | Smooth blend nodule patch into CT (not used for spiculated). |
| `get_location` | `self, ct_array, lung_mask, region` | Return anatomical coordinates for insertion. |
| `visualize_fake_nodule_region` | `self, ct_array, location, size` | Show a cropped CT region containing the synthetic nodule. |
| `visualize_real_nodule_region` | `self, ct_array, nodule_bbox, size, index` | Show a cropped CT region of real nodule. |
| `ct_pipeline` | `self, nodule, region, size, alpha` | Master pipeline to insert a synthetic nodule into CT. |

---


## 📄 `helper_functions.py`

This module contains all visualization utilities, shape generation routines, geometry computations, and noise-based morphing operations used during nodule or tissue synthesis.

### Functions

- **`vis(array, size, title)`**: Visualize a 3D image slice-by-slice using a slider (`SimpleITK` array).
- **`play_volume(image, interval, in_, out_, loop)`**: Simulate video playback of 3D volume slices.
- **`visualize_base(points, title)`**: Interactive 3D point cloud visualization of a surface.
- **`visualize_boundaries_with_slider(boundary_points, volume_shape)`**: Slice-wise viewer for 3D boundaries using a slider.
- **`vis_overlay(base, overlay, size, title, alpha, cmap_overlay, axis)`**: Overlay a 3D hollowness or mask map onto a base CT using a slider viewer.

### Shape Generation Utilities

- **`generate_sphere(num_pts, image_size, radius)`**: Generate 3D sphere surface points in a volume.
- **`generate_ellipsoid(num_pts, image_size, rx, ry, rz)`**: Generate ellipsoid surface points with radii along each axis.
- **`generate_superellipsoid(num_pts, image_size, rx, ry, rz, eps1, eps2)`**: Generate a generalized ellipsoid with rounded or boxy features.
- **`generate_radial_noise_shape(num_pts, image_size)`**: Add low-frequency noise to a circular shape for realistic boundaries.

### Geometry + Shape Analysis

- **`shrink_tissue(vol, new_shape)`**: Resize/shrink volume to a new shape (no docstring).
- **`get_radius_for_theta(mask, theta, max_radius)`**: For a given direction `θ`, return max radius for a binary mask.
- **`compute_compactness_2d(binary_mask)`**: Calculate 2D compactness: `4π·Area / Perimeter²`.
- **`compute_compactness_3d(binary_mask)`**: Compute compactness of a 3D object (no docstring).
- **`find_flow_params(target_compactness, tissue_type, nodule_type, tolerance)`**: Return deformation flow parameters based on compactness constraints (no docstring).

### Morphology + Mask Conversions

- **`boundary_to_mask(boundary_points, volume_shape)`**: Convert surface point cloud to 3D binary mask.
- **`insert_holes_random(volume, euler_num, sphere_r, hole_r)`**: Add holes to a volume randomly using Perlin noise.
- **`insert_holes_random_distant(volume, euler_num, sphere_r, hole_r, max_attempts)`**: Insert non-overlapping holes inside a sphere.
- **`compute_hollowness(volume, sigma)`**: Compute a hollowness intensity map using Gaussian filtering (no docstring).
- **`find_k_separated_minima(hollow_map, k, min_distance, smooth_sigma)`**: Return `k` most hollow, well-separated regions (used for placement).
- **`extract_all_surfaces(volume)`**: Return all connected surfaces in a binary mask (no docstring).
- **`split_boundary_into_holes(boundary_points, eps, min_samples)`**: Use DBSCAN to segment boundary points into individual hole surfaces.


## 📓 `example.ipynb`

No functions or classes defined. Presumably demonstrates usage of the above modules.

---



## 🔁 Cross-Usage Highlights

- `Nodule.tissue_pipeline()` and `Nodule.nodule_pipeline()` are master generation flows.
- `CTScan.ct_pipeline()` merges output from above pipelines into a real CT.
- Functions from `helper_functions.py` are used across `nodule.py` for shape manipulation and insertion logic.

---



# Task 2

# Binary Lung Cancer Classifier (Chest CT) — Ensemble of CNN Backbones

This repository/notebook implements a **binary classifier** that predicts whether a chest CT study indicates **Normal** or **Cancerous**. The approach centers on a **strong, diversity-driven ensemble** of convolutional backbones that balances representation power, generalization, and clinical robustness.

> Notebook: `Binary_Classifier_Final_(Normal_Cancerous).ipynb`
> Task: Binary classification — *Normal* vs *Cancerous* (lung cancer prediction)

---

## Why an Ensemble Is the Right Tool for This Task

Chest CTs are heterogeneous across scanners, kernels, and patient habitus. Cancer patterns vary from obvious solid nodules to faint ground-glass opacities. No single model family consistently dominates across all these regimes. An **ensemble**:

* **Increases robustness** to acquisition variability and windowing differences by aggregating models with different inductive biases.
* **Improves calibration** of predicted probabilities by averaging logits (reducing overconfident errors).
* **Captures complementary features**: texture sensitivity from VGG-style stacks, multi-scale efficiency from EfficientNet, and deep residual abstraction from ResNet.
* **Reduces variance** and tail risk, which matters in clinically oriented decision support.

---

## Model Architecture

### Backbone Trio (VER-style Ensemble)

1. **VGG19-BN**

   * Deep, sequential conv blocks with batch norm.
   * Strength: **fine texture patterns** (e.g., GGO, subtle spiculation).
   * Modification: final classifier head replaced by `Linear(in_features, 2)`.

2. **EfficientNet-B0**

   * MBConv + compound scaling for parameter efficiency.
   * Strength: **multi-scale context** with strong accuracy/latency trade-off.
   * Modification: final classifier layer replaced by `Linear(in_features, 2)`.

3. **ResNet101**

   * Deep residual blocks alleviate vanishing gradients.
   * Strength: **high-level semantic abstraction** and stable optimization.
   * Modification: `fc` replaced by `Linear(in_features, 2)`.

> All three are **ImageNet-pretrained** for improved convergence and generalization, then **fine-tuned** for the lung cancer task.

### Fusion Strategy

* **Logit Averaging (Late Fusion)**  
  Each backbone outputs raw logits \( z_i \in \mathbb{R}^2 \).  
  The ensemble combines them as:

  $$
  z_{\text{ens}} = \frac{1}{N}\sum_{i=1}^{N} z_i \quad \text{and} \quad p = \text{softmax}(z_{\text{ens}})
  $$

  This keeps decision boundaries flexible while avoiding catastrophic votes from any single model.

* (Optional) **Weighted Logit Averaging**
  If validation indicates different strengths, you can set weights `w_i` (∑w_i=1) to emphasize the best backbone.

### Why This Architecture Fits Lung CT Classification

* **Texture + Structure**: VGG19-BN excels at **micro-textures**; ResNet101 excels at **macro-structures**; EfficientNet-B0 balances both with **scaling efficiency**.
* **Heterogeneity-tolerant**: Combining different receptive-field dynamics mitigates shifts across scanners and reconstruction kernels.
* **Calibration-aware**: Late-fusion of logits improves reliability of confidence—important when thresholding for clinic-facing triage.
* **Extensible**: You can slot in 3D backbones or hybrid 2.5D inputs later without changing the fusion interface.

---

## Training Pipeline (Notebook)

* **Loss**: Cross-Entropy (binary task with two logits).
* **Optimizer**: Adam/SGD with cosine or step LR (configurable in the notebook).
* **Transfer Learning**:

  * Phase 1: Freeze most layers; train heads for rapid alignment.
  * Phase 2: Unfreeze progressively; fine-tune with a lower LR.
* **Augmentations** (recommended & implemented in-notebook):

  * Geometric: small rotations, flips (within anatomical reason), random crops/resize.
  * Intensity: subtle brightness/contrast jitter; optional CLAHE; mild Gaussian noise.
  * Medical-aware: **Hounsfield windowing** into lung windows if using CT intensities, then normalize.
* **Imbalance Handling**:

  * Class weights in Cross-Entropy **or** focal loss variant (optional).
  * Stratified sampling per epoch.

> The notebook includes cells to toggle/adjust augmentations, LR schedulers, and freezing strategies.

---

## Evaluation

* **Primary metrics**: AUROC, AUPRC, Accuracy, F1, Sensitivity (Recall of *Cancerous*), Specificity.
* **Thresholding**: Pick operating points based on ROC/PR curves; in clinical screening, **high sensitivity** is often prioritized.
* **Calibration**: Reliability curves / Expected Calibration Error (ECE) recommended; ensemble typically improves this.
* **Ablations**: The notebook provides per-backbone metrics and “leave-one-out” ensemble checks to verify contribution.

---

## Interpretability (Recommended)

* **Class Activation Mapping (e.g., Grad-CAM)** over suspected lesions to ensure the model attends to plausible lung regions.
* **Error Review**: Provide confusion breakdowns by slice view or study-level aggregates if you use multiple slices/projections.

> Interpretability is essential to build trust and to surface spurious correlations.

---

## Reproducibility

Set seeds and deterministic flags where practical:

```python
import torch, numpy as np, random, os
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
```

---

## Environment & Requirements

* **Python**: 3.10+
* **Core**: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`
* **Augmentation** (optional): `albumentations`
* **Medical I/O** (if you load raw CT volumes): `pydicom` or `nibabel`, `scikit-image`

Install (example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install numpy pandas scikit-learn matplotlib albumentations
pip install pydicom nibabel scikit-image  # if using DICOM/NIfTI flows
```

---

## How to Use

1. **Open the notebook**
   `Binary_Classifier_Final_(Normal_Cancerous).ipynb` in Jupyter/VS Code.

2. **Configure paths & settings**

   * Point the data loader to your preprocessed CT images or projections.
   * Choose backbones to include in the ensemble (all three by default).
   * Set training hyperparameters (epochs, LR, batch size).

3. **Run training**
   Execute cells sequentially. The notebook will:

   * Build each backbone with a 2-class head
   * Train with class weighting / augmentations (if enabled)
   * Track metrics per-backbone and for the ensemble

4. **Evaluate & export**

   * Generate ROC/PR plots, confusion matrix, and calibration plots
   * Save best checkpoints and (optionally) a TorchScript bundle for deployment

---




