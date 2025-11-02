# Stage 3: CT Description Pipeline (CT-CLIP)

## Overview

This stage uses CT-CLIP (Contrastive Language-Image Pre-training for Computed Tomography) models to analyze chest CT scans and detect 30+ lung cancer-related pathologies.

## Models

### Available Models

1. **CT-CLIP Base v2** (`ct_clip_v2.pt`)
   - Size: 1.7GB
   - AUROC: 0.772
   - Best for: General chest CT analysis

2. **CT-CLIP VocabFine v2** (`ct_vocabfine_v2.pt`)
   - Size: 1.7GB
   - AUROC: 0.824
   - Best for: Pathology-specific detection (recommended)

3. **Ensemble** (Base 40% + VocabFine 60%)

### Model Download

```bash
cd ../models

# Download CT-CLIP Base
wget https://huggingface.co/[model-url]/ct_clip_v2.pt

# Download CT-CLIP VocabFine
wget https://huggingface.co/[model-url]/ct_vocabfine_v2.pt
```

Or use the download script:
```bash
python download_models.py
```

## Detected Pathologies

### Standard 18 Pathologies (CT-CLIP)

1. Medical material
2. Arterial wall calcification
3. Cardiomegaly
4. Pericardial effusion
5. Coronary artery wall calcification
6. Hiatal hernia
7. Lymphadenopathy ⭐
8. Emphysema ⭐
9. Atelectasis ⭐
10. Lung nodule ⭐ (PRIMARY)
11. Lung opacity ⭐ (PRIMARY)
12. Pulmonary fibrotic sequela ⭐
13. Pleural effusion ⭐
14. Mosaic attenuation pattern
15. Peribronchial thickening
16. Consolidation ⭐
17. Bronchiectasis
18. Interlobular septal thickening

⭐ = Lung cancer relevant

### Additional 12 Lung Cancer Terms (Custom Queries)

1. Mass (>3cm lesion)
2. Spiculated nodule (high cancer suspicion)
3. Ground glass opacity (early adenocarcinoma)
4. Cavitary lesion
5. Post-obstructive pneumonia
6. Superior vena cava obstruction
7. Apical mass (Pancoast tumor)
8. Mediastinal mass
9. Pleural thickening
10. Pleural nodularity
11. Chest wall invasion
12. Satellite nodules

## Usage

### 1. Single Model Inference

```python
from ct_clip_inference import CTClipInferenceSingle

# Initialize
inferencer = CTClipInferenceSingle(
    model_path="../models/ct_vocabfine_v2.pt"
)

# Run inference
predictions = inferencer.infer(
    volume_path="../data/sample_ct.nii.gz"
)

print(predictions)
# Output: {'Lung nodule': 0.68, 'Emphysema': 0.52, ...}
```

### 2. Ensemble Inference (Recommended)

```python
from ensemble_inference import run_ensemble_inference

# Run ensemble (Base 40% + VocabFine 60%)
predictions = run_ensemble_inference(
    volume_path="../data/sample_ct.nii.gz",
    base_model_path="../models/ct_clip_v2.pt",
    vocabfine_model_path="../models/ct_vocabfine_v2.pt"
)

print(predictions)
```

### 3. Generate Lung Cancer-Focused Report

```python
from lung_cancer_report_generator import generate_lung_cancer_focused_report

# Generate report
report = generate_lung_cancer_focused_report(
    predictions=predictions,
    model_name="Ensemble (VocabFine + Base)",
    scan_info={
        'filename': 'patient_123_ct.nii.gz',
        'analyzed_at': '2025-11-02 15:30:00'
    },
    include_custom_terms=True
)

print(report)
# Saves to: patient_123_ct_lung_cancer_report.txt
```

### 4. Command Line Usage

```bash
# Single model
python ct_clip_inference.py \
    --model_path ../models/ct_vocabfine_v2.pt \
    --input_ct ../data/sample_ct.nii.gz \
    --output results.json

# Ensemble
python ensemble_inference.py \
    --input_ct ../data/sample_ct.nii.gz \
    --output results.json

# Generate report
python lung_cancer_report_generator.py results.json
```

## Report Structure

The lung cancer-focused report includes:

```
═══════════════════════════════════════════════════════
       LUNG CANCER SCREENING REPORT
═══════════════════════════════════════════════════════

Model: Ensemble (VocabFine + Base)
Scan File: patient_123_ct.nii.gz
Analysis Date: 2025-11-02 15:30:00

───────────────────────────────────────────────────────
SECTION 1: LUNG CANCER RELATED FINDINGS
───────────────────────────────────────────────────────

PRIMARY CONCERNS (High Risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Lung Nodule**
  • Status: [o]  DETECTED (confidence: 68%)
  • Explanation: A small round growth in the lung tissue...
  • Risk Level: High
  • Recommended Action: Follow-up imaging and possible biopsy

 ASSOCIATED FINDINGS (Medium Risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Lymphadenopathy**
  • Status: [o] DETECTED (confidence: 45%)
  • Explanation: Enlarged lymph nodes in the chest...
  • Risk Level: Medium
  • Recommended Action: Monitor for cancer spread

RISK FACTORS (Increases Cancer Risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Emphysema** 
  • Status: [o] DETECTED (confidence: 52%)
  • Explanation: Lung damage from smoking...
  • Risk Level: Low (but increases risk)
  • Recommended Action: Smoking cessation critical

───────────────────────────────────────────────────────
SECTION 2: OTHER MEDICAL FINDINGS
───────────────────────────────────────────────────────

- Heart & Vascular
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Cardiomegaly** 
  • Status: [x] DETECTED (confidence: 67%)
  • Explanation: Enlarged heart (NOT lung cancer related)
  • Category: Cardiac condition
  • Recommended Action: Cardiology evaluation

───────────────────────────────────────────────────────
SECTION 3: ADDITIONAL LUNG CANCER SCREENING TERMS
───────────────────────────────────────────────────────

Other important patterns to look for:
  • Ground glass opacity: Hazy areas (early cancer sign)
  • Spiculated nodule: Irregular borders (40-80% cancer)
  • Satellite nodules: Small nodules near main nodule
  ...

───────────────────────────────────────────────────────
SECTION 4: CLINICAL SUMMARY & RECOMMENDATIONS
───────────────────────────────────────────────────────

OVERALL ASSESSMENT: FINDINGS REQUIRE MEDICAL ATTENTION

Critical findings detected that may indicate lung cancer.

IMMEDIATE ACTIONS:
  1. Consult pulmonologist within 48-72 hours
  2. Consider PET-CT for further evaluation
  3. Gather previous CT scans for comparison
  4. Biopsy may be necessary for nodule characterization

═══════════════════════════════════════════════════════
```

## Performance Metrics

| Model | AUROC | Inference Time | GPU Memory |
|-------|-------|----------------|------------|
| Base | 0.772 | ~1.5s | ~4GB |
| VocabFine | 0.824 | ~1.5s | ~4GB |

## Input Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Size**: Any (tested up to 2GB)
- **Modality**: Non-contrast chest CT
- **Orientation**: Any (automatically reoriented)

## Output Formats

1. **JSON**: Machine-readable predictions
   ```json
   {
     "Lung nodule": 0.68,
     "Emphysema": 0.52,
     "Cardiomegaly": 0.67,
     ...
   }
   ```

2. **Text Report**: Human-readable lung cancer-focused report
3. **CSV**: Tabular format for batch processing

## Customization

### Adjust Thresholds

Edit `lung_cancer_report_generator.py`:

```python
PATHOLOGY_THRESHOLDS = {
    'Lung nodule': 0.25,  # Lower = more sensitive
    'Mass': 0.30,
    'Emphysema': 0.40,
    # ...
}
```

### Add Custom Terms

```python
ADDITIONAL_LUNG_CANCER_TERMS = {
    'your_new_term': {
        'category': 'Primary Concern',
        'explanation': 'Plain English explanation',
        'risk_level': 'High',
        'action': 'Recommended action'
    }
}
```

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.0.0`
- `nibabel>=5.0.0`
- `einops>=0.7.0`
- `transformers>=4.30.0`
- `timm>=0.9.0`

## Troubleshooting

**Issue**: CUDA out of memory
```python
# Use CPU instead
device = 'cpu'
```

**Issue**: Slow inference
```python
# Use smaller batch size or VocabFine only (not ensemble)
```

**Issue**: Model file not found
```bash
# Check model path
ls -lh ../models/
```

## References

1. CT-CLIP paper: [arXiv link]
2. CT-RATE dataset: [dataset link]
3. Fleischner Society Guidelines: [clinical guidelines]

---

**Ready to analyze CT scans! 🫁**
