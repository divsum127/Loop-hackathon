"""
Model Setup for CT-CLIP Inference

This module handles the initialization and loading of the CT-CLIP model components.
"""

import torch
from transformers import BertTokenizer, BertModel


def initialize_ct_clip_model(model_path=None, device='cuda'):
    """
    Initialize CT-CLIP model components.
    
    Args:
        model_path: Path to pretrained CT-CLIP model weights
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Dictionary containing model, tokenizer, and device
    """
    try:
        # Import CT-CLIP specific modules
        # Note: These should be installed from the CT-CLIP repository
        from transformer_maskgit import CTViT
        from ct_clip import CTCLIP
    except ImportError:
        raise ImportError(
            "CT-CLIP modules not found. Please install from:\n"
            "git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git\n"
            "cd CT-CLIP/transformer_maskgit && pip install -e .\n"
            "cd .. && pip install -e ."
        )
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    print(f"Initializing CT-CLIP model on {device}...")
    
    # Initialize BERT tokenizer and text encoder
    print("Loading BERT tokenizer and text encoder...")
    tokenizer = BertTokenizer.from_pretrained(
        'microsoft/BiomedVLP-CXR-BERT-specialized',
        do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize CT image encoder (CTViT)
    print("Initializing CT Vision Transformer...")
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )
    
    # Initialize CT-CLIP model
    print("Initializing CT-CLIP model...")
    clip_model = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )
    
    # Load pretrained weights if provided
    if model_path:
        print(f"Loading pretrained weights from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Remove incompatible keys
            keys_to_remove = []
            for key in state_dict.keys():
                if 'position_ids' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                print(f"Removing incompatible key: {key}")
                del state_dict[key]
            
            # Load with strict=False to ignore missing keys
            missing_keys, unexpected_keys = clip_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys (will use random initialization): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                
            print("✓ Model weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Continuing with randomly initialized weights...")
    else:
        print("No model path provided. Using randomly initialized weights.")
    
    # Move model to device
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    return {
        'model': clip_model,
        'tokenizer': tokenizer,
        'device': device
    }


def get_pathology_list():
    """
    Returns the list of 18 pathologies that CT-CLIP can detect.
    
    Returns:
        List of pathology names
    """
    return [
        'Medical material',
        'Arterial wall calcification',
        'Cardiomegaly',
        'Pericardial effusion',
        'Coronary artery wall calcification',
        'Hiatal hernia',
        'Lymphadenopathy',
        'Emphysema',
        'Atelectasis',
        'Lung nodule',
        'Lung opacity',
        'Pulmonary fibrotic sequela',
        'Pleural effusion',
        'Mosaic attenuation pattern',
        'Peribronchial thickening',
        'Consolidation',
        'Bronchiectasis',
        'Interlobular septal thickening'
    ]


def prepare_text_prompts(pathology_name):
    """
    Prepare positive and negative text prompts for a pathology.
    
    Args:
        pathology_name: Name of the pathology
    
    Returns:
        List of two strings: [positive_prompt, negative_prompt]
    """
    positive = f"{pathology_name} is present."
    negative = f"{pathology_name} is not present."
    return [positive, negative]


if __name__ == "__main__":
    # Test model initialization
    print("Testing CT-CLIP model initialization...")
    
    try:
        model_dict = initialize_ct_clip_model(device='cpu')
        print("\nModel initialized successfully!")
        print(f"Device: {model_dict['device']}")
        print(f"Model type: {type(model_dict['model'])}")
        print(f"Tokenizer type: {type(model_dict['tokenizer'])}")
        
        # Test pathology list
        pathologies = get_pathology_list()
        print(f"\nNumber of pathologies: {len(pathologies)}")
        print("Pathologies:", pathologies[:3], "...")
        
    except Exception as e:
        print(f"\nError during initialization: {e}")
        print("\nPlease ensure CT-CLIP repository is installed.")
