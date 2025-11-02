#!/usr/bin/env python3
"""
Simple report generator that converts CT-CLIP predictions (probabilities) into
a human-readable radiology-style narrative.

Usage:
    python report_generator.py results/valid_1_a_1.nii_20251102_053701.json

Outputs:
    - A TXT report saved alongside the JSON results.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Basic mapping of pathologies to report sections and phrasing
PATHOLOGY_MAP = {
    'Medical material': ('Mediastinum', 'Medical material or devices noted'),
    'Arterial wall calcification': ('Vascular', 'Calcific plaques observed in the arterial walls'),
    'Cardiomegaly': ('Heart', 'Heart size is enlarged (cardiomegaly)'),
    'Pericardial effusion': ('Pericardium', 'Pericardial effusion present'),
    'Coronary artery wall calcification': ('Vascular', 'Coronary artery wall calcifications noted'),
    'Hiatal hernia': ('Gastroesophageal', 'Hiatal hernia noted'),
    'Lymphadenopathy': ('Mediastinum', 'Enlarged mediastinal or hilar lymph nodes'),
    'Emphysema': ('Lungs', 'Emphysematous changes in the lungs'),
    'Atelectasis': ('Lungs', 'Linear or subsegmental atelectasis present'),
    'Lung nodule': ('Lungs', 'Lung nodule(s) detected'),
    'Lung opacity': ('Lungs', 'Lung opacity / consolidation present'),
    'Pulmonary fibrotic sequela': ('Lungs', 'Fibrotic sequela in the lungs'),
    'Pleural effusion': ('Pleura', 'Pleural effusion present'),
    'Mosaic attenuation pattern': ('Lungs', 'Mosaic attenuation pattern observed'),
    'Peribronchial thickening': ('Lungs', 'Peribronchial thickening and bronchial wall thickening'),
    'Consolidation': ('Lungs', 'Lobar/segmental consolidation noted'),
    'Bronchiectasis': ('Lungs', 'Bronchiectatic changes present'),
    'Interlobular septal thickening': ('Lungs', 'Interlobular septal thickening observed')
}

# Default thresholds for mapping probabilities to confidence levels
DEFAULT_THRESHOLDS = {
    'high': 0.7,
    'medium': 0.5,
    'low': 0.3
}

# Example pathology-specific thresholds (lower for lung nodule)
PATHOLOGY_SPECIFIC_THRESHOLDS = {
    'Lung nodule': {'high': 0.6, 'medium': 0.4, 'low': 0.25},
    # You can add more overrides here
}

# Helper to convert probability into a confidence phrase
def confidence_phrase(prob, thresholds=DEFAULT_THRESHOLDS):
    if prob >= thresholds['high']:
        return 'definite', 'There is'  # strong language
    elif prob >= thresholds['medium']:
        return 'probable', 'There is likely'
    elif prob >= thresholds['low']:
        return 'possible', 'There may be'
    else:
        return 'unlikely', 'No significant'

# High-level section order
SECTION_ORDER = ['Lungs', 'Pleura', 'Mediastinum', 'Vascular', 'Heart', 'Pericardium', 'Gastroesophageal', 'Abdomen', 'Bones']

# Generate sentences per section

def generate_report(results: dict, thresholds=None, pathology_specific_thresholds=None, paragraph_mode=False):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if pathology_specific_thresholds is None:
        pathology_specific_thresholds = PATHOLOGY_SPECIFIC_THRESHOLDS
    predictions = results.get('predictions', {})

    # Gather findings by section
    sections = {s: [] for s in SECTION_ORDER}
    incidental = []

    for pathology, prob in predictions.items():
        mapping = PATHOLOGY_MAP.get(pathology)
        if not mapping:
            incidental.append((pathology, prob))
            continue
        section, phrase = mapping

        # allow pathology-specific thresholds
        local_thresholds = thresholds
        if pathology in pathology_specific_thresholds:
            local_thresholds = pathology_specific_thresholds[pathology]

        conf, lead = confidence_phrase(prob, local_thresholds)
        sentence = ''
        if conf == 'unlikely':
            # skip adding negative statements for each pathology, we'll add normal statements later
            continue
        # Customize phrasing for common patterns
        if pathology == 'Arterial wall calcification':
            sentence = f"{lead} {phrase.lower()} (probability {prob:.2f})."
        elif pathology == 'Pleural effusion':
            sentence = f"{lead} pleural effusion (probability {prob:.2f})."
        elif pathology == 'Cardiomegaly':
            sentence = f"{lead} {phrase.lower()} (probability {prob:.2f})."
        elif pathology == 'Lung nodule':
            sentence = f"{lead} {phrase.lower()} detected; consider correlation with dedicated chest CT or follow-up (probability {prob:.2f})."
        else:
            sentence = f"{lead} {phrase.lower()} (probability {prob:.2f})."

        # Append this pathology sentence into the appropriate section
        sections.setdefault(section, []).append((prob, sentence))

    # Build narrative
    narrative = []
    header = f"Automated CT-CLIP Report\nGenerated: {datetime.utcnow().isoformat()} UTC\nSource: {results.get('volume_name','unknown')}\n"
    narrative.append(header)

    if paragraph_mode:
        # Build a single-paragraph summary aiming for the style the user provided
        para_parts = []
        # Vascular/mediastinal/heart
        vascular_items = sorted(sections.get('Vascular', []), key=lambda x: x[0], reverse=True)
        mediastinal_items = sorted(sections.get('Mediastinum', []), key=lambda x: x[0], reverse=True)
        heart_items = sorted(sections.get('Heart', []), key=lambda x: x[0], reverse=True)
        pericardium_items = sorted(sections.get('Pericardium', []), key=lambda x: x[0], reverse=True)

        if mediastinal_items:
            # e.g., venous collaterals / lymphadenopathy / medical material
            for p, s in mediastinal_items:
                para_parts.append(s)
        if vascular_items:
            for p, s in vascular_items:
                para_parts.append(s)
        if heart_items:
            for p, s in heart_items:
                para_parts.append(s)
        if pericardium_items:
            for p, s in pericardium_items:
                para_parts.append(s)

        # Lungs and pleura combined
        lung_items = sorted(sections.get('Lungs', []), key=lambda x: x[0], reverse=True)
        pleural_items = sorted(sections.get('Pleura', []), key=lambda x: x[0], reverse=True)
        if lung_items:
            # group common lung sentences into a combined sentence
            lung_phrases = [s.rstrip('.') for _, s in lung_items]
            combined_lung = ' '.join(lung_phrases) + '.'
            para_parts.append(combined_lung)
        if pleural_items:
            pleura_phrases = [s.rstrip('.') for _, s in pleural_items]
            combined_pleura = ' '.join(pleura_phrases) + '.'
            para_parts.append(combined_pleura)

        # Abdomen/incidental - not mapped in predictions, but include if present
        if sections.get('Abdomen'):
            for p, s in sections.get('Abdomen'):
                para_parts.append(s)

        # Combine into a paragraph
        paragraph = ' '.join(para_parts)
        narrative.append('\nFINDINGS:\n')
        narrative.append(paragraph + '\n\n')

        # Impression: top 6
        narrative.append('IMPRESSION:\n')
        top_findings = sorted([(p, k) for k, p in predictions.items()], reverse=True)[:6]
        for prob, pathology in top_findings:
            # use pathology-specific thresholds if available
            local_thresholds = thresholds
            if pathology in pathology_specific_thresholds:
                local_thresholds = pathology_specific_thresholds[pathology]
            conf, _ = confidence_phrase(prob, local_thresholds)
            narrative.append(f"- {pathology}: {conf} (probability {prob:.2f})\n")

        return '\n'.join(narrative)

    # Non-paragraph (list) mode follows original behavior
    narrative.append('\nFINDINGS:\n')

    # Lungs and pleura first
    for section in SECTION_ORDER:
        items = sections.get(section, [])
        if not items:
            continue
        # sort by probability desc
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        narrative.append(f"{section}:\n")
        for prob, sent in items_sorted:
            narrative.append(f"  - {sent}\n")
        narrative.append('\n')

    # Add normal statements for sections with no findings
    narrative.append('IMPRESSION:\n')
    # Generate impression lines from top findings
    # Take up to 6 top findings
    top_findings = sorted([(p, k) for k, p in predictions.items()], reverse=True)[:6]
    for prob, pathology in top_findings:
        conf, lead = confidence_phrase(prob, thresholds)
        if conf == 'definite':
            narrative.append(f"- {pathology}: {conf} (probability {prob:.2f})\n")
        elif conf == 'probable':
            narrative.append(f"- {pathology}: {conf} (probability {prob:.2f})\n")
        elif conf == 'possible':
            narrative.append(f"- {pathology}: {conf} (probability {prob:.2f})\n")

    # Incidental low-confidence notes (if any)
    if incidental:
        narrative.append('\nIncidental findings (unmapped pathologies):\n')
        for p, prob in incidental:
            narrative.append(f"- {p}: probability {prob:.2f}\n")

    return '\n'.join(narrative)


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a radiology-style report from CT-CLIP JSON results')
    parser.add_argument('results_json', help='Path to the CT-CLIP results JSON file')
    parser.add_argument('--paragraph', action='store_true', help='Emit a single-paragraph FINDINGS section')
    parser.add_argument('--high', type=float, help='Override high confidence threshold (default: 0.7)')
    parser.add_argument('--medium', type=float, help='Override medium confidence threshold (default: 0.5)')
    parser.add_argument('--low', type=float, help='Override low confidence threshold (default: 0.3)')
    parser.add_argument('--no-path-thresholds', action='store_true', help='Disable pathology-specific threshold overrides')
    args = parser.parse_args()

    results_file = Path(args.results_json)
    if not results_file.exists():
        print(f'File not found: {results_file}')
        sys.exit(1)
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Build thresholds dict (use defaults unless overridden)
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.high is not None:
        thresholds['high'] = args.high
    if args.medium is not None:
        thresholds['medium'] = args.medium
    if args.low is not None:
        thresholds['low'] = args.low

    pathology_thresholds = None if args.no_path_thresholds else PATHOLOGY_SPECIFIC_THRESHOLDS

    report_text = generate_report(results, thresholds=thresholds, pathology_specific_thresholds=pathology_thresholds, paragraph_mode=args.paragraph)
    out_file = results_file.with_suffix('.report.txt')
    with open(out_file, 'w') as f:
        f.write(report_text)
    print(f'Report saved to {out_file}')
    print('\n')
    print(report_text)
