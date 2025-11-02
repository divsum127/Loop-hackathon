"""
CT-CLIP Inference for Single Volume

Main inference script for running CT-CLIP model on a single CT volume.
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import warnings

from .data_loader import CTVolumeLoader, CTRATEMetadataLoader
from .model_setup import (
    initialize_ct_clip_model,
    get_pathology_list,
    prepare_text_prompts
)


class CTClipInferenceSingle:
    """
    CT-CLIP inference engine for single volume analysis.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Initialize CT-CLIP inference engine.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Classification threshold for binary predictions
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        
        # Initialize model components
        print("=" * 60)
        print("Initializing CT-CLIP Inference Engine")
        print("=" * 60)
        
        model_dict = initialize_ct_clip_model(
            model_path=model_path,
            device=self.device
        )
        
        self.model = model_dict['model']
        self.tokenizer = model_dict['tokenizer']
        self.pathologies = get_pathology_list()
        
        # Initialize volume loader
        self.volume_loader = CTVolumeLoader()
        
        print(f"\n✓ Model ready on {self.device}")
        print(f"✓ Will detect {len(self.pathologies)} pathologies")
        print(f"✓ Classification threshold: {self.threshold}")
    
    def apply_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax to logits.
        
        Args:
            logits: Input logits
        
        Returns:
            Softmax probabilities
        """
        softmax = torch.nn.Softmax(dim=0)
        return softmax(logits)
    
    def infer_single_pathology(
        self,
        volume_tensor: torch.Tensor,
        pathology_name: str
    ) -> float:
        """
        Run inference for a single pathology.
        
        Args:
            volume_tensor: Preprocessed CT volume tensor
            pathology_name: Name of the pathology to detect
        
        Returns:
            Probability that the pathology is present (0-1)
        """
        # Prepare text prompts
        text_prompts = prepare_text_prompts(pathology_name)
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Move volume to device
        volume_tensor = volume_tensor.to(self.device)
        
        # Run model inference
        with torch.no_grad():
            output = self.model(
                text_tokens,
                volume_tensor,
                device=self.device
            )
        
        # Apply softmax to get probabilities
        probs = self.apply_softmax(output)
        
        # Return probability of presence (first element)
        return probs[0].cpu().item()
    
    def infer_all_pathologies(
        self,
        volume_tensor: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run inference for all pathologies.
        
        Args:
            volume_tensor: Preprocessed CT volume tensor
            verbose: Whether to print progress
        
        Returns:
            Dictionary mapping pathology names to probabilities
        """
        predictions = {}
        
        if verbose:
            print(f"\nRunning inference for {len(self.pathologies)} pathologies...")
        
        for i, pathology in enumerate(self.pathologies, 1):
            if verbose:
                print(f"  [{i}/{len(self.pathologies)}] {pathology}...", end=' ')
            
            try:
                prob = self.infer_single_pathology(volume_tensor, pathology)
                predictions[pathology] = prob
                
                if verbose:
                    status = "✓ POSITIVE" if prob >= self.threshold else "✗ negative"
                    print(f"{prob:.4f} {status}")
                    
            except Exception as e:
                import traceback
                if verbose:
                    print(f"ERROR:")
                    traceback.print_exc()
                predictions[pathology] = -1.0  # Error indicator
        
        return predictions
    
    def infer_single_volume(
        self,
        volume_path: str,
        metadata_file: Optional[str] = None,
        volume_name: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Complete inference pipeline for a single CT volume.
        
        Args:
            volume_path: Path to CT volume (.nii.gz file)
            metadata_file: Optional path to metadata CSV
            volume_name: Optional volume name for metadata lookup
            verbose: Whether to print progress
        
        Returns:
            Dictionary containing all inference results
        """
        start_time = datetime.now()
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"CT-CLIP Inference: {Path(volume_path).name}")
            print("=" * 60)
        
        # Load and preprocess volume
        volume_tensor, volume_metadata = self.volume_loader.load_and_preprocess(
            volume_path,
            return_metadata=True
        )
        
        # Load additional metadata if available
        dataset_metadata = None
        if metadata_file and volume_name:
            metadata_loader = CTRATEMetadataLoader(metadata_path=metadata_file)
            dataset_metadata = metadata_loader.get_volume_metadata(volume_name)
        
        # Run inference
        predictions = self.infer_all_pathologies(volume_tensor, verbose=verbose)
        
        # Identify positive findings
        positive_findings = [
            pathology for pathology, prob in predictions.items()
            if prob >= self.threshold
        ]
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Compile results
        results = {
            'volume_name': Path(volume_path).name,
            'volume_path': str(volume_path),
            'timestamp': datetime.now().isoformat(),
            'inference_time_seconds': inference_time,
            'device': self.device,
            'threshold': self.threshold,
            'predictions': predictions,
            'positive_findings': positive_findings,
            'num_positive': len(positive_findings),
            'volume_metadata': volume_metadata,
            'dataset_metadata': dataset_metadata
        }
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Inference completed in {inference_time:.2f} seconds")
            print(f"Positive findings: {len(positive_findings)}/{len(self.pathologies)}")
            if positive_findings:
                print("\nDetected pathologies:")
                for pathology in positive_findings:
                    prob = predictions[pathology]
                    print(f"  • {pathology}: {prob:.4f}")
            else:
                print("\nNo significant pathologies detected.")
            print("=" * 60)
        
        return results
    
    def save_results(
        self,
        results: Dict,
        output_dir: str,
        formats: List[str] = ['json', 'txt', 'csv']
    ):
        """
        Save inference results to files.
        
        Args:
            results: Results dictionary from infer_single_volume
            output_dir: Directory to save results
            formats: List of output formats ('json', 'txt', 'csv')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        volume_name = Path(results['volume_name']).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        if 'json' in formats:
            json_path = output_path / f"{volume_name}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Saved JSON: {json_path}")
        
        # Save text report
        if 'txt' in formats:
            txt_path = output_path / f"{volume_name}_{timestamp}.txt"
            with open(txt_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("CT-CLIP Inference Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Volume: {results['volume_name']}\n")
                f.write(f"Date: {results['timestamp']}\n")
                f.write(f"Inference Time: {results['inference_time_seconds']:.2f}s\n")
                f.write(f"Device: {results['device']}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("FINDINGS\n")
                f.write("-" * 60 + "\n\n")
                
                if results['positive_findings']:
                    f.write(f"Detected {len(results['positive_findings'])} pathologies:\n\n")
                    for pathology in results['positive_findings']:
                        prob = results['predictions'][pathology]
                        f.write(f"  • {pathology}: {prob:.4f}\n")
                else:
                    f.write("No significant pathologies detected.\n")
                
                f.write("\n" + "-" * 60 + "\n")
                f.write("ALL PREDICTIONS\n")
                f.write("-" * 60 + "\n\n")
                
                for pathology, prob in results['predictions'].items():
                    status = "POSITIVE" if prob >= self.threshold else "negative"
                    f.write(f"{pathology:.<45} {prob:.4f} ({status})\n")
            
            print(f"✓ Saved report: {txt_path}")
        
        # Save CSV
        if 'csv' in formats:
            import pandas as pd
            csv_path = output_path / f"{volume_name}_{timestamp}.csv"
            
            df_data = []
            for pathology, prob in results['predictions'].items():
                df_data.append({
                    'Pathology': pathology,
                    'Probability': prob,
                    'Prediction': 'Positive' if prob >= self.threshold else 'Negative'
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV: {csv_path}")


if __name__ == "__main__":
    # Example usage
    print("CT-CLIP Inference Module")
    print("=" * 60)
    print("\nThis module provides inference capabilities for CT-CLIP.")
    print("Use run_inference.py for command-line inference.")
