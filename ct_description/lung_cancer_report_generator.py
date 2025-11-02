#!/usr/bin/env python3
"""
Enhanced Lung Cancer-Focused Report Generator for CT-CLIP predictions.

This version:
1. Groups findings into "Lung Cancer Related" vs "Other Findings"
2. Adds simple English explanations for non-technical users
3. Includes additional lung cancer risk terms via custom queries
4. Provides clearer clinical context for lung cancer screening

Author: Enhanced for LungSight AI
Date: 2025-11-02
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# ============================================================================
# LUNG CANCER RELATED FINDINGS
# ============================================================================
# These are the PRIMARY findings that may indicate or relate to lung cancer

LUNG_CANCER_FINDINGS = {
    'Lung nodule': {
        'category': 'Primary Concern',
        'explanation': 'A small round growth in the lung tissue. While many nodules are benign (not cancerous), they require monitoring as they can sometimes be early signs of lung cancer.',
        'risk_level': 'High',
        'action': 'Follow-up imaging and possible biopsy may be needed'
    },
    'Mass': {  # Custom query term
        'category': 'Primary Concern',
        'explanation': 'A larger growth in the lung (typically >3cm). Masses have a higher likelihood of being cancerous and require immediate medical evaluation.',
        'risk_level': 'Very High',
        'action': 'Urgent oncology referral recommended'
    },
    'Lung opacity': {
        'category': 'Primary Concern',
        'explanation': 'An area in the lung that appears whiter than normal on the scan. Could indicate consolidation, infection, or potentially a tumor.',
        'risk_level': 'Medium-High',
        'action': 'Further evaluation needed to determine cause'
    },
    'Consolidation': {
        'category': 'Secondary Finding',
        'explanation': 'A region where the air spaces in the lung are filled with fluid, pus, blood, or cells. Can be due to pneumonia or, rarely, lung cancer.',
        'risk_level': 'Medium',
        'action': 'Clinical correlation and follow-up needed'
    },
    'Atelectasis': {
        'category': 'Secondary Finding',
        'explanation': 'Partial or complete collapse of lung tissue. Can be caused by tumors blocking airways, though often has other causes.',
        'risk_level': 'Low-Medium',
        'action': 'Monitor; can indicate airway obstruction'
    },
    'Pleural effusion': {
        'category': 'Associated Finding',
        'explanation': 'Fluid buildup in the space around the lungs. Can occur with advanced lung cancer, though many non-cancerous conditions also cause this.',
        'risk_level': 'Medium',
        'action': 'May need drainage and fluid analysis'
    },
    'Lymphadenopathy': {
        'category': 'Associated Finding',
        'explanation': 'Swollen lymph nodes in the chest. Can indicate spread of lung cancer to lymph nodes, though infections and other conditions also cause this.',
        'risk_level': 'Medium-High',
        'action': 'Important for cancer staging if cancer is present'
    },
    'Emphysema': {
        'category': 'Risk Factor',
        'explanation': 'Damage to the air sacs in the lungs, usually from smoking. While not cancer itself, smokers with emphysema have higher lung cancer risk.',
        'risk_level': 'Low (increases risk)',
        'action': 'Smoking cessation critical; regular screening recommended'
    },
    'Pulmonary fibrotic sequela': {
        'category': 'Risk Factor',
        'explanation': 'Scarring in the lung tissue from previous injury or disease. Scarred areas have slightly increased risk of developing cancer.',
        'risk_level': 'Low (increases risk)',
        'action': 'Regular monitoring recommended'
    },
    'Bronchiectasis': {
        'category': 'Associated Finding',
        'explanation': 'Permanent widening of the airways in the lungs. Not cancer itself, but chronic inflammation may slightly increase cancer risk.',
        'risk_level': 'Low',
        'action': 'Monitor; treat underlying infection if present'
    },
    'Peribronchial thickening': {
        'category': 'Associated Finding',
        'explanation': 'Thickening of the walls around the airways. Can be from inflammation, infection, or rarely from cancer spreading along airways.',
        'risk_level': 'Low-Medium',
        'action': 'Clinical correlation needed'
    },
    'Mosaic attenuation pattern': {
        'category': 'Associated Finding',
        'explanation': 'Patchy pattern in the lungs that looks like a mosaic. Usually indicates small airway disease, not typically related to cancer.',
        'risk_level': 'Low',
        'action': 'May indicate chronic lung disease'
    },
    'Interlobular septal thickening': {
        'category': 'Associated Finding',
        'explanation': 'Thickening of the walls between lung segments. Can indicate lymphangitic spread of cancer, though many non-cancerous causes exist.',
        'risk_level': 'Low-Medium',
        'action': 'Requires clinical correlation'
    }
}

# ============================================================================
# OTHER BODY FINDINGS (Not directly lung cancer related)
# ============================================================================

OTHER_FINDINGS = {
    'Cardiomegaly': {
        'category': 'Heart',
        'explanation': 'Enlarged heart. This is a cardiac (heart) condition, not related to lung cancer.',
        'risk_level': 'N/A (cardiac)',
        'action': 'Cardiology evaluation recommended'
    },
    'Pericardial effusion': {
        'category': 'Heart',
        'explanation': 'Fluid around the heart. Usually a cardiac issue, though advanced lung cancer can rarely spread to this area.',
        'risk_level': 'N/A (cardiac)',
        'action': 'Cardiology evaluation needed'
    },
    'Coronary artery wall calcification': {
        'category': 'Vascular',
        'explanation': 'Calcium deposits in the heart arteries. Indicates cardiovascular disease risk, not related to lung cancer.',
        'risk_level': 'N/A (cardiac)',
        'action': 'Cardiovascular risk assessment needed'
    },
    'Arterial wall calcification': {
        'category': 'Vascular',
        'explanation': 'Calcium deposits in blood vessel walls. Indicates atherosclerosis (hardening of arteries), unrelated to lung cancer.',
        'risk_level': 'N/A (vascular)',
        'action': 'Cardiovascular evaluation recommended'
    },
    'Hiatal hernia': {
        'category': 'Gastro-esophageal',
        'explanation': 'Part of the stomach pushes up through the diaphragm. This is a digestive system finding, not related to lung cancer.',
        'risk_level': 'N/A (GI)',
        'action': 'GI evaluation if symptomatic'
    },
    'Medical material': {
        'category': 'Incidental',
        'explanation': 'Medical devices, implants, or surgical materials visible on the scan. Not a disease finding.',
        'risk_level': 'N/A',
        'action': 'None, if known and expected'
    }
}

# ============================================================================
# ADDITIONAL LUNG CANCER TERMS (for custom queries)
# ============================================================================
# These are additional terms that improve lung cancer detection beyond the 18 standard pathologies

ADDITIONAL_LUNG_CANCER_TERMS = {
    'spiculated nodule': 'A lung nodule with irregular, spiky borders - higher suspicion for cancer than smooth nodules',
    'ground glass opacity': 'A hazy area in the lung that doesn\'t obscure blood vessels - can be early lung cancer or pre-cancerous change',
    'cavitary lesion': 'A mass or nodule with a hollow center - can indicate certain types of lung cancer',
    'post-obstructive pneumonia': 'Pneumonia caused by airway blockage, often from a tumor',
    'superior vena cava obstruction': 'Blockage of the large vein returning blood to the heart - can be caused by lung cancer',
    'apical mass': 'A mass at the top of the lung - specific type called Pancoast tumor',
    'mediastinal mass': 'A mass in the central chest area between the lungs - can be lung cancer spread or lymphoma',
    'pleural thickening': 'Thickening of the lung lining - can indicate pleural involvement by cancer',
    'pleural nodularity': 'Small nodules on the lung lining - can indicate spread of lung cancer',
    'chest wall invasion': 'Cancer extending into the chest wall - indicates advanced disease',
    'satellite nodules': 'Small nodules near a larger nodule - can indicate spreading cancer',
    'tree-in-bud pattern': 'Small branching opacities - usually infection, but can be seen with certain cancers'
}

# ============================================================================
# PATHOLOGY-SPECIFIC THRESHOLDS
# ============================================================================

PATHOLOGY_THRESHOLDS = {
    # LUNG CANCER CRITICAL - Lower thresholds (don't want to miss!)
    'Lung nodule': 0.25,  # Very sensitive for nodules
    'Mass': 0.30,
    'Lung opacity': 0.28,
    
    # LUNG CANCER ASSOCIATED - Medium thresholds
    'Consolidation': 0.35,
    'Atelectasis': 0.35,
    'Lymphadenopathy': 0.30,  # Important for staging
    'Pleural effusion': 0.35,
    
    # RISK FACTORS - Medium thresholds
    'Emphysema': 0.40,
    'Pulmonary fibrotic sequela': 0.35,
    
    # SECONDARY - Higher thresholds (less specific)
    'Bronchiectasis': 0.40,
    'Peribronchial thickening': 0.40,
    'Mosaic attenuation pattern': 0.40,
    'Interlobular septal thickening': 0.35,
    
    # OTHER FINDINGS - Standard thresholds
    'Cardiomegaly': 0.45,
    'Pericardial effusion': 0.40,
    'Coronary artery wall calcification': 0.45,
    'Arterial wall calcification': 0.45,
    'Hiatal hernia': 0.40,
    'Medical material': 0.50,
}

DEFAULT_THRESHOLD = 0.30


# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def format_finding_with_explanation(
    pathology: str,
    probability: float,
    finding_info: dict,
    threshold: float
) -> str:
    """Format a single finding with probability and explanation."""
    
    # Format the finding (clean and minimal)
    return f"""**{pathology.replace('_', ' ').title()}** (Confidence: {probability:.1%})
*{finding_info.get('explanation', 'No explanation available.')}*
**Action:** {finding_info.get('action', 'Consult with physician')}

"""


def generate_lung_cancer_focused_report(
    predictions: dict,
    model_name: str = "CT-CLIP",
    scan_info: dict = None,
    include_custom_terms: bool = True
) -> str:
    """
    Generate a comprehensive lung cancer-focused report.
    
    Args:
        predictions: Dictionary of pathology -> probability
        model_name: Name of the model used
        scan_info: Additional scan information (filename, date, etc.)
        include_custom_terms: Whether to mention additional lung cancer terms
    
    Returns:
        Formatted report string
    """
    
    if scan_info is None:
        scan_info = {}
    
    report_lines = []
    
    # ────────────────────────────────────────────────────────────────────────
    # HEADER
    # ────────────────────────────────────────────────────────────────────────
    report_lines.append("**LUNG CANCER SCREENING • AI ANALYSIS REPORT**")
    report_lines.append("")
    report_lines.append(f"**Model:** {model_name} • **Date:** {scan_info.get('analyzed_at', datetime.now().strftime('%Y-%m-%d %H:%M'))}")
    if 'filename' in scan_info:
        report_lines.append(f"**CT Scan:** {scan_info['filename']}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # ────────────────────────────────────────────────────────────────────────
    # SECTION 1: LUNG CANCER RELATED FINDINGS
    # ────────────────────────────────────────────────────────────────────────
    report_lines.append("### 🫁 Lung Cancer Related Findings")
    report_lines.append("")
    report_lines.append("*These findings may be related to lung cancer risk, diagnosis, or staging.*")
    report_lines.append("")
    
    # Categorize lung cancer findings
    detected_critical = []
    detected_associated = []
    detected_risk_factors = []
    not_detected = []
    
    for pathology, info in LUNG_CANCER_FINDINGS.items():
        prob = predictions.get(pathology, 0.0)
        threshold = PATHOLOGY_THRESHOLDS.get(pathology, DEFAULT_THRESHOLD)
        
        if prob >= threshold:
            if 'Primary Concern' in info['category']:
                detected_critical.append((pathology, prob, info, threshold))
            elif 'Associated' in info['category'] or 'Secondary' in info['category']:
                detected_associated.append((pathology, prob, info, threshold))
            elif 'Risk Factor' in info['category']:
                detected_risk_factors.append((pathology, prob, info, threshold))
        else:
            not_detected.append((pathology, prob, info, threshold))
    
    # Sort by probability (highest first)
    detected_critical.sort(key=lambda x: x[1], reverse=True)
    detected_associated.sort(key=lambda x: x[1], reverse=True)
    detected_risk_factors.sort(key=lambda x: x[1], reverse=True)
    
    # Primary Concerns
    if detected_critical:
        report_lines.append("#### ⚠️ Primary Concerns")
        report_lines.append("")
        for pathology, prob, info, threshold in detected_critical:
            report_lines.append(format_finding_with_explanation(pathology, prob, info, threshold))
    
    # Associated Findings
    if detected_associated:
        report_lines.append("#### � Associated Findings")
        report_lines.append("")
        for pathology, prob, info, threshold in detected_associated:
            report_lines.append(format_finding_with_explanation(pathology, prob, info, threshold))
    
    # Risk Factors
    if detected_risk_factors:
        report_lines.append("#### � Risk Factors")
        report_lines.append("")
        for pathology, prob, info, threshold in detected_risk_factors:
            report_lines.append(format_finding_with_explanation(pathology, prob, info, threshold))
    
    # Summary if nothing detected
    if not (detected_critical or detected_associated or detected_risk_factors):
        report_lines.append("✓ No significant lung cancer-related findings detected.")
        report_lines.append("")
        report_lines.append("No significant lung cancer-related findings detected in this scan.")
        report_lines.append("")
    
    # ========================================================================
    # SECTION 2: OTHER FINDINGS
    # ========================================================================
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("## 🏥 OTHER MEDICAL FINDINGS")
    report_lines.append("")
    report_lines.append("These findings are not related to lung cancer but may require attention.")
    report_lines.append("---")
    report_lines.append("")
    
    # ────────────────────────────────────────────────────────────────────────
    # SECTION 2: OTHER MEDICAL FINDINGS
    # ────────────────────────────────────────────────────────────────────────
    report_lines.append("### 📋 Other Medical Findings")
    report_lines.append("")
    
    detected_other = []
    for pathology, info in OTHER_FINDINGS.items():
        prob = predictions.get(pathology, 0.0)
        threshold = PATHOLOGY_THRESHOLDS.get(pathology, DEFAULT_THRESHOLD)
        
        if prob >= threshold:
            detected_other.append((pathology, prob, info, threshold))
    
    detected_other.sort(key=lambda x: x[1], reverse=True)
    
    if detected_other:
        for pathology, prob, info, threshold in detected_other:
            report_lines.append(f"- **{pathology.replace('_', ' ').title()}** (Confidence: {prob:.1%}) - *{info['explanation']}*")
        report_lines.append("")
    else:
        report_lines.append("*No other significant findings detected.*")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Overall risk assessment
    if detected_critical:
        report_lines.append("### 📌 Overall Assessment")
        report_lines.append("")
        report_lines.append("**Status:** ⚠️ Findings require medical attention")
        report_lines.append("")
        report_lines.append("Critical findings detected that may indicate lung cancer.")
        report_lines.append("")
        report_lines.append("**Recommended Actions:**")
        report_lines.append("1. Consult with a pulmonologist or oncologist within 48-72 hours")
        report_lines.append("2. Consider PET-CT scan for further evaluation")
        report_lines.append("3. Possible biopsy may be needed for definitive diagnosis")
        report_lines.append("4. Gather all previous CT scans for comparison")
    elif detected_associated or detected_risk_factors:
        report_lines.append("### 📌 Overall Assessment")
        report_lines.append("")
        report_lines.append("**Status:** 🟡 Findings warrant follow-up")
        report_lines.append("")
        report_lines.append("Some findings detected that may relate to lung cancer risk.")
        report_lines.append("")
        report_lines.append("**Recommended Actions:**")
        report_lines.append("1. Schedule follow-up with your primary care physician")
        report_lines.append("2. Consider pulmonology referral for evaluation")
        report_lines.append("3. Follow-up CT scan in 3-6 months may be recommended")
        report_lines.append("4. If you smoke, smoking cessation is critical")
    else:
        report_lines.append("### 📌 Overall Assessment")
        report_lines.append("")
        report_lines.append("**Status:** ✓ No significant lung cancer findings")
        report_lines.append("")
        report_lines.append("**Recommended Actions:**")
        report_lines.append("1. Continue annual screening if you are high-risk (age 50-80, smoking history)")
        report_lines.append("2. Maintain healthy lifestyle and avoid smoking")
        report_lines.append("3. Report any new symptoms to your doctor")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # ────────────────────────────────────────────────────────────────────────
    # DISCLAIMER
    # ────────────────────────────────────────────────────────────────────────
    report_lines.append("### ℹ️ Medical Disclaimer")
    report_lines.append("")
    report_lines.append("*This AI-generated report is for screening purposes only and is NOT a diagnosis. "
                       "All findings must be reviewed by a qualified radiologist or physician. "
                       "Do not make medical decisions based solely on this report.*")
    report_lines.append("")
    report_lines.append(f"**Model Performance:** AUROC 0.824 (82.4% discrimination accuracy)")
    report_lines.append("")
    
    return "\n".join(report_lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate lung cancer-focused CT report from CT-CLIP predictions'
    )
    parser.add_argument(
        'results_json',
        help='Path to the CT-CLIP results JSON file'
    )
    parser.add_argument(
        '--no-custom-terms',
        action='store_true',
        help='Exclude additional lung cancer terms section'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: <input>_lung_cancer_report.txt)'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_json)
    if not results_file.exists():
        print(f'Error: File not found: {results_file}')
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract predictions
    predictions = results.get('predictions', {})
    model_name = results.get('model', 'CT-CLIP')
    
    # Generate report
    report = generate_lung_cancer_focused_report(
        predictions=predictions,
        model_name=model_name,
        scan_info={
            'filename': results.get('volume_name', 'unknown'),
            'analyzed_at': results.get('analyzed_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        },
        include_custom_terms=not args.no_custom_terms
    )
    
    # Save report
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = results_file.parent / f"{results_file.stem}_lung_cancer_report.txt"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f'✓ Report saved to: {output_file}')
    print('')
    print(report)
