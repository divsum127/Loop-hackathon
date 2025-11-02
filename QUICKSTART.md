# Quick Start Guide - LungSight AI

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
cd submission
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Models
```bash
cd models
python download_models.py
cd ..
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
```

### 4. Run Application
```bash
streamlit run app.py
```

Open browser at **http://localhost:8501**

---

## 📖 Usage Workflow

### Step 1: Upload CT Scan
- Click **"Browse files"** in sidebar
- Upload `.nii.gz` CT scan (up to 2GB)
- Select model: **Ensemble** (recommended)

### Step 2: Configure Patient Profile
- Enter patient information:
  - Age, gender
  - Smoking history
  - Occupation
  - Medical history
  - Symptoms

### Step 3: Analyze
- Click **"Analyze CT Scan"** button
- Wait 5-10 seconds for processing

### Step 4: Review Results
- **Lung Cancer Findings**: Primary concerns, associated findings, risk factors
- **Other Findings**: Cardiac, vascular, GI
- **Risk Assessment**: Overall cancer risk score
- **Recommendations**: Personalized action plan

### Step 5: Ask Questions
- Use chat interface for follow-up questions
- Get clarification on medical terms
- Discuss next steps

---

## 📁 Sample Data

Test with provided sample:
```bash
# Sample CT scan location
data/sample_ct_scans/sample_patient_001.nii.gz

# Patient profile for testing
Age: 62
Gender: Male
Smoking: Former smoker (30 pack-years)
Occupation: Construction worker
History: COPD
```

---

## 🎯 Key Features

### Model Selection
- **Ensemble**: Best accuracy (AUROC 0.87) ✅ Recommended
- **VocabFine**: Fast & accurate (AUROC 0.824)
- **Base**: General analysis (AUROC 0.772)

### Report Sections
1. 🫁 **Lung Cancer Related Findings**
   - 🔴 Primary Concerns (high risk)
   - 🟡 Associated Findings (medium risk)
   - 🟠 Risk Factors (increases risk)

2. 🏥 **Other Medical Findings**
   - Heart & vascular conditions
   - GI findings

3. 📋 **Additional Screening Terms**
   - 12 specialized lung cancer patterns

4. 📊 **Clinical Summary**
   - Overall assessment
   - Recommended actions
   - Follow-up schedule

### Urgency Levels
- 🔴 **CRITICAL** (80-100): 24-48 hour action
- 🟠 **HIGH** (60-79): 48-72 hour follow-up
- 🟡 **MODERATE** (40-59): 1-2 week follow-up
- 🟢 **LOW** (0-39): Routine screening

---

## 💡 Tips

### For Best Results
1. Use **Ensemble model** for maximum accuracy
2. Provide complete patient history
3. Upload high-quality CT scans (512×512+ resolution)
4. Review all sections of the report

### Common Questions
**Q: How long does analysis take?**
A: 5-10 seconds with GPU, 15-20 seconds with CPU

**Q: What CT scan formats are supported?**
A: NIfTI (.nii, .nii.gz)

**Q: Can I analyze multiple scans?**
A: Yes, upload one at a time or use batch processing

**Q: Is patient data stored?**
A: No, all processing is in-memory (privacy-first)

---

## 🔧 Troubleshooting

### App won't start
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall streamlit
pip install streamlit

# Run with explicit python
python -m streamlit run app.py
```

### Upload fails
```bash
# Check file size (max 2GB)
ls -lh your_ct_scan.nii.gz

# Verify file format
file your_ct_scan.nii.gz
# Should show: gzip compressed data
```

### Slow inference
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### OpenAI API errors
```bash
# Verify API key
cat .env | grep OPENAI_API_KEY

# Check quota at https://platform.openai.com/usage
```

---

## 📚 Documentation

- **Full README**: `README.md`
- **Installation Guide**: `docs/INSTALLATION.md`
- **Stage 3 (CT Analysis)**: `ct_description/README.md`
- **Stage 4 (Recommendations)**: `health_recommendations/README.md`

---

## 🎉 Example Output

```
═══════════════════════════════════════════════════════
       LUNG CANCER SCREENING REPORT
═══════════════════════════════════════════════════════

🫁 LUNG CANCER RELATED FINDINGS

🔴 PRIMARY CONCERNS

**Lung Nodule** 🟠
  • Status: ✓ DETECTED (confidence: 68%)
  • Explanation: A small round growth in the lung tissue.
    Many are benign, but require monitoring.
  • Risk Level: High
  • Action: Follow-up imaging and possible biopsy needed

---

📊 CLINICAL SUMMARY

⚠️ OVERALL ASSESSMENT: HIGH RISK (Risk Score: 78/100)

IMMEDIATE ACTIONS:
  1. Consult pulmonologist within 48-72 hours
  2. Schedule PET-CT scan
  3. Consider biopsy for nodule characterization

═══════════════════════════════════════════════════════
```

---

## 🎓 For Hackathon Judges

### Testing the System

1. **Upload sample CT**: `data/sample_ct_scans/sample_patient_001.nii.gz`
2. **Use test profile**: See "Sample Data" section above
3. **Expected result**: Lung nodule detected with high-risk classification
4. **Processing time**: ~7 seconds (GPU) or ~15 seconds (CPU)

### Key Innovations
- ✅ 4-stage end-to-end pipeline
- ✅ 30+ pathology detection (vs. standard 18)
- ✅ Lung cancer-focused reports with plain English
- ✅ AI-powered personalized recommendations
- ✅ Interactive chat for patient questions

### Performance Metrics
- **Stage 2**: 92.4% accuracy, 0.96 AUC-ROC (nodule classification)
- **Stage 3**: 0.87 AUC-ROC (ensemble CT description)
- **Stage 4**: <5s response time (LangGraph recommendations)
- **End-to-End**: ~10-15s total processing

---

## 📞 Support

- **Email**: [your-email]
- **Issues**: See `docs/INSTALLATION.md`
- **Demo video**: [link if available]

---

**Ready to analyze CT scans and save lives! 🫁💙**
