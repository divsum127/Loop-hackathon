"""
Data Loading Utilities for CT-CLIP Inference

Handles loading and preprocessing of CT volumes from the CT-RATE dataset.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from pathlib import Path


class CTVolumeLoader:
    """
    Loader for CT volumes in NIfTI format with preprocessing.
    """
    
    def __init__(
        self,
        target_shape=(200, 480, 480),  # (D, H, W) - depth must be divisible by temporal_patch_size=10
        normalize=True,
        hu_min=-1000,
        hu_max=1000
    ):
        """
        Initialize CT volume loader.
        
        Args:
            target_shape: Target shape for resizing (height, width, depth)
            normalize: Whether to apply intensity normalization
        """
        self.target_shape = target_shape
        self.normalize = normalize
    
    def load_nifti(self, file_path):
        """
        Load a NIfTI file and return the volume array.
        
        Args:
            file_path: Path to .nii or .nii.gz file
        
        Returns:
            Tuple of (volume_array, affine, header)
        """
        print(f"Loading NIfTI file: {file_path}")
        
        try:
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header
            
            print(f"Original shape: {volume.shape}")
            print(f"Data type: {volume.dtype}")
            print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")
            
            return volume, affine, header
            
        except Exception as e:
            raise RuntimeError(f"Error loading NIfTI file {file_path}: {e}")
    
    def normalize_intensity(self, volume, window_center=-600, window_width=1500):
        """
        Apply intensity normalization (windowing) to CT volume.
        
        Args:
            volume: Input CT volume
            window_center: HU window center
            window_width: HU window width
        
        Returns:
            Normalized volume in range [0, 1]
        """
        # Apply CT windowing
        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2
        
        volume = np.clip(volume, min_value, max_value)
        volume = (volume - min_value) / (max_value - min_value)
        
        return volume
    
    def resize_volume(self, volume, target_shape):
        """
        Resize volume to target shape using trilinear interpolation.
        
        Args:
            volume: Input volume
            target_shape: Target shape (H, W, D)
        
        Returns:
            Resized volume
        """
        print(f"Resizing volume from {volume.shape} to {target_shape}...")
        
        # Calculate zoom factors
        zoom_factors = [
            target_shape[i] / volume.shape[i] 
            for i in range(3)
        ]
        
        # Resize using scipy
        resized = ndimage.zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def preprocess_volume(self, volume):
        """
        Apply full preprocessing pipeline to CT volume.
        
        Args:
            volume: Input CT volume
        
        Returns:
            Preprocessed volume ready for model input
        """
        # Normalize intensity
        if self.normalize:
            volume = self.normalize_intensity(volume)
        
        # Resize to target shape
        if volume.shape != self.target_shape:
            volume = self.resize_volume(volume, self.target_shape)
        
        return volume
    
    def prepare_batch(self, volume):
        """
        Convert volume to PyTorch tensor with batch dimension.
        
        Args:
            volume: Preprocessed volume array
        
        Returns:
            PyTorch tensor with shape (1, 1, H, W, D)
        """
        # Add channel and batch dimensions
        volume_tensor = torch.from_numpy(volume).float()
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        
        return volume_tensor
    
    def load_and_preprocess(self, file_path, return_metadata=False):
        """
        Complete pipeline: load, preprocess, and prepare CT volume.
        
        Args:
            file_path: Path to NIfTI file
            return_metadata: Whether to return metadata
        
        Returns:
            PyTorch tensor ready for inference, optionally with metadata
        """
        # Load NIfTI file
        volume, affine, header = self.load_nifti(file_path)
        
        # Preprocess
        volume = self.preprocess_volume(volume)
        
        # Prepare batch
        volume_tensor = self.prepare_batch(volume)
        
        if return_metadata:
            metadata = {
                'original_shape': header.get_data_shape(),
                'spacing': header.get_zooms(),
                'affine': affine,
                'preprocessed_shape': volume.shape,
                'file_path': str(file_path)
            }
            return volume_tensor, metadata
        
        return volume_tensor


class CTRATEMetadataLoader:
    """
    Loader for CT-RATE dataset metadata and reports.
    """
    
    def __init__(self, metadata_path=None, reports_path=None):
        """
        Initialize metadata loader.
        
        Args:
            metadata_path: Path to metadata CSV file
            reports_path: Path to reports CSV file
        """
        self.metadata_path = metadata_path
        self.reports_path = reports_path
        self.metadata_df = None
        self.reports_df = None
        
        if metadata_path and Path(metadata_path).exists():
            self.metadata_df = pd.read_csv(metadata_path)
            print(f"Loaded metadata: {len(self.metadata_df)} entries")
        
        if reports_path and Path(reports_path).exists():
            self.reports_df = pd.read_csv(reports_path)
            print(f"Loaded reports: {len(self.reports_df)} entries")
    
    def get_volume_metadata(self, volume_name):
        """
        Get metadata for a specific volume.
        
        Args:
            volume_name: Name of the volume (e.g., 'valid_1_a_1')
        
        Returns:
            Dictionary with metadata or None if not found
        """
        if self.metadata_df is None:
            return None
        
        # Try to find matching entry
        matches = self.metadata_df[
            self.metadata_df['VolumeName'] == volume_name
        ]
        
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        
        return None
    
    def get_volume_report(self, volume_name):
        """
        Get radiology report for a specific volume.
        
        Args:
            volume_name: Name of the volume
        
        Returns:
            Dictionary with report text or None if not found
        """
        if self.reports_df is None:
            return None
        
        matches = self.reports_df[
            self.reports_df['VolumeName'] == volume_name
        ]
        
        if len(matches) > 0:
            report = matches.iloc[0]
            return {
                'findings': report.get('Findings_EN', ''),
                'impressions': report.get('Impressions_EN', '')
            }
        
        return None


if __name__ == "__main__":
    # Test volume loader
    print("Testing CT Volume Loader...")
    
    loader = CTVolumeLoader()
    print(f"Initialized loader with target shape: {loader.target_shape}")
    
    # Create a dummy volume for testing
    dummy_volume = np.random.randn(512, 512, 100) * 100 - 600
    print(f"\nDummy volume shape: {dummy_volume.shape}")
    
    # Test preprocessing
    preprocessed = loader.preprocess_volume(dummy_volume)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Value range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
    
    # Test batch preparation
    batch = loader.prepare_batch(preprocessed)
    print(f"Batch tensor shape: {batch.shape}")
    
    print("\n✓ Data loader test complete!")
