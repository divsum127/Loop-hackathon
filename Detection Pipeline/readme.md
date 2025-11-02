
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
  Each backbone outputs raw logits `z_i ∈ ℝ²`. We compute
  [
  z_{\text{ens}} = \frac{1}{N}\sum_{i=1}^{N} z_i \quad \text{and} \quad p = \text{softmax}(z_{\text{ens}})
  ]
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




