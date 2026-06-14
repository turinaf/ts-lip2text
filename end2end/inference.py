import torch
import cv2
import numpy as np
from models.verification_model import LipTextVerificationModel
from utils.phoneme_converter import PhonemeConverter
from utils.dataset_utils import Config
import argparse
import sys
import os
import glob
import re

def parse_input_text(text, dataset):
    if dataset == 'grid':
        words = []
        # Handle literal '\n' from command line
        lines = text.replace('\\n', '\n').strip().split('\n')
        if len(lines) > 1 or (len(lines) == 1 and len(lines[0].split()) == 3 and lines[0].split()[0].isdigit()):
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    if word not in ['sil', 'sp']:
                        words.append(word)
            return " ".join(words)
        else:
            # Also filter 'sil' and 'sp' if plain text is provided
            return " ".join([w for w in text.split() if w not in ['sil', 'sp']])
    else:
        return text

def resolve_checkpoint_path(requested_model_path, config_output_dir):
    if requested_model_path and os.path.exists(requested_model_path):
        return requested_model_path

    preferred = [
        os.path.join(config_output_dir, "best_lip_verification_model.pth"),
        os.path.join(config_output_dir, "best_lip_verification_model.pt"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    patterns = [
        "checkpoint_ep*.pt",
        "checkpoint_ep*.pth",
        "checkpoint_epoch_*.pt",
        "checkpoint_epoch_*.pth",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(config_output_dir, pattern)))

    if not candidates:
        return None

    def extract_epoch(path):
        name = os.path.basename(path)
        m = re.search(r"(?:checkpoint_ep|checkpoint_epoch_)(\d+)", name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=extract_epoch)
    return candidates[-1]


def run_inference(video_path, transcript, model_path=None, dataset=None):
    cfg = Config("config.json")
    dataset = dataset or cfg.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    proj_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.abspath(os.path.join(proj_root, cfg.output_dir))
    resolved_model_path = resolve_checkpoint_path(model_path, output_dir_abs)

    if resolved_model_path is None:
        print(f"No checkpoint found. Checked requested path and output dir: {output_dir_abs}")
        sys.exit(1)

    print(f"Loading Model from {resolved_model_path} onto {device}...")
    
    transcript = parse_input_text(transcript, dataset)
    
    vocab_size = 13 if dataset == "digit" else 41
    # Needs to match architecture in main
    model = LipTextVerificationModel(
        vocab_size=vocab_size,
        feature_dim=256,
        lip_backbone_pretrained=cfg.lip_backbone_pretrained
    ).to(device)

    checkpoint = torch.load(resolved_model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    print(f"Parsing transcript: '{transcript}'")
    if dataset == "digit":
        digit_vocab = {str(i): i+1 for i in range(10)} 
        digit_vocab["!"] = 11
        digit_vocab["<pad>"] = 0
        digit_vocab["<unk>"] = 12
        phones = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in transcript.split()]
    else:
        converter = PhonemeConverter()
        _, phones = converter.text_to_phonemes(transcript)
        
    if len(phones) == 0:
        phones = [0]

    # Keep a bounded sequence length to match training regime while retaining more transcript context.
    max_phone_len = 128
    phones = phones[:max_phone_len]
    phone_tensor = torch.tensor(phones, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
    phoneme_padding_mask = phone_tensor.eq(0)

    print(f"Loading specific video frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Assuming crop matches your 96x96 preprocessing requirement
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (96, 96))
        frames.append(frame)
        
    cap.release()
    if len(frames) == 0:
        print("Failed to load video frames.")
        sys.exit(1)

    max_frames = cfg.max_frames
    true_video_len = min(len(frames), max_frames)
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        last = frames[-1]
        frames.extend([last] * (max_frames - len(frames)))

    frames = np.stack(frames)
    # Models expect shape: (Batch, Time, Channels, Height, Width)
    # frames is (Time, Height, Width, Channels)
    video_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    # Apply the same normalization as train/test: T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    video_tensor = (video_tensor - 0.5) / 0.5
    video_tensor = video_tensor.unsqueeze(0).to(device)
    video_padding_mask = (torch.arange(max_frames, device=device).unsqueeze(0) >= true_video_len)

    print(f"Evaluating alignment...")
    with torch.no_grad():
        score, align_out = model(
            video_tensor,
            phone_tensor,
            audio_features=None,
            video_padding_mask=video_padding_mask,
            phoneme_padding_mask=phoneme_padding_mask
        )

    # Output Confidence 
    prob = score.squeeze().item()
    # result = "MATCH" if prob > 0.5 else "MISMATCH"
    
    print(f"=====================================")
    # print(f"Result: {result}")
    print(f"Alignment score (Probability): {prob:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip-to-Text Inference Verification")
    parser.add_argument("--video", type=str, required=True, help="Path to cropped MP4 lip file")
    parser.add_argument("--text", type=str, required=True, help="Ground truth or hypothesized text")
    parser.add_argument("--model", type=str, default=None, help="Optional checkpoint model path; if omitted, uses latest from config output_dir")
    parser.add_argument("--dataset", type=str, default=None, choices=["grid", "digit"], help="Optional dataset override; defaults to config dataset")
    
    args = parser.parse_args()
    run_inference(args.video, args.text, args.model, args.dataset)
