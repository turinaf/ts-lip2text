import torch
import os
import glob
import re
from torch.utils.data import DataLoader
from dataset.grid_loader import GRIDDataset
from models.verification_model import LipTextVerificationModel
from evaluation.evaluate import evaluate_model
from utils.dataset_utils import collate_fn_factory, Config
from dataset.digit_loader import DigitDataset


def _resolve_latest_checkpoint(output_dir_abs):
    # Priority order: explicit best checkpoint, then latest epoch checkpoint.
    preferred = [
        os.path.join(output_dir_abs, "best_lip_verification_model.pth"),
        os.path.join(output_dir_abs, "best_lip_verification_model.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    patterns = [
        "checkpoint_ep*.pt",       # current training format
        "checkpoint_ep*.pth",
        "checkpoint_epoch_*.pth",  # legacy format
        "checkpoint_epoch_*.pt",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(output_dir_abs, pattern)))

    if not candidates:
        return None

    def extract_epoch(path):
        name = os.path.basename(path)
        match = re.search(r"(?:checkpoint_ep|checkpoint_epoch_)(\d+)", name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=extract_epoch)
    return candidates[-1]

def test():
    cfg = Config("config.json")
    proj_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.abspath(os.path.join(proj_root, cfg.output_dir))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = os.path.abspath(os.path.join(proj_root, cfg.processed_data_dir))
    
    if not os.path.exists(dataset_path):
        print(f"Error: Processed dataset directory ({dataset_path}) is missing.")
        return
        
    print(f"Loading test dataset from {dataset_path}...")
    if cfg.dataset == "digit":
        test_dataset = DigitDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    else:
        test_dataset = GRIDDataset(data_dir=dataset_path, split='test', max_frames=cfg.max_frames, use_audio=cfg.use_audio)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty. Check your data splits.")
        return
        
    print(f"Test split loaded -> {len(test_dataset)} samples")
    
    # Standard PyTorch DataLoader for evaluation
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_factory(cfg.dataset)
    )
    
    print("Initializing model...")
    if cfg.dataset == "digit":
        vocab_size = 13  # 0-9, !, <pad>, <unk>
    else:
        vocab_size = 41  # PhonemeConverter vocab used in training
    model = LipTextVerificationModel(
        vocab_size=vocab_size,
        feature_dim=256,
        lip_backbone_pretrained=cfg.lip_backbone_pretrained
    ).to(device)
    
    model_save_path = _resolve_latest_checkpoint(output_dir_abs)
    if model_save_path is None:
        print(f"No checkpoints found under {output_dir_abs}. Please train the model first.")
        return
    
    print(f"Loading weights from {model_save_path}...")
    checkpoint = torch.load(model_save_path, map_location=device)
    
    # Handle both standalone weights and wrapped checkpoint dictionaries
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    # Handle DDP weights (stripping the 'module.' prefix if the model was trained with Multi-GPU)
    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        # Remap legacy single-alignment keys to lip_text alignment
        if key.startswith("alignment."):
            key = key.replace("alignment.", "alignment_lip_text.", 1)
        clean_state_dict[key] = v
            
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print(f"  Note: {len(missing)} new keys initialised from scratch (model architecture was extended).")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in checkpoint (ignored).")
    
    print("\n[--- Testing Phase ---]")
    print("Evaluating Model on Hold-out Test Set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nFinal Test Result -> AUC: {test_metrics['roc_auc']:.4f} | Acc: {test_metrics['verification_accuracy']:.4f}")

if __name__ == "__main__":
    test()
