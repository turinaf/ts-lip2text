import os
import glob
import torch
import cv2
import json
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as T

# Mel-spectrogram parameters — must match AudioEncoder's n_mels expectation
_MEL_SAMPLE_RATE = 16000
_MEL_N_MELS = 80
_MEL_N_FFT = 400
_MEL_HOP_LENGTH = 200

class GRIDDataset(Dataset):
    def __init__(self, data_dir, split='train', max_frames=75, transform=None, use_audio=False):
        """
        Loads the preprocessed GRID dataset using pre-computed split files.
        data_dir should point to the processed directory (e.g. ./data) containing 
        the train_split.json, val_split.json, and test_split.json manifests.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.use_audio = use_audio
        self.transform = transform or T.Compose([
            T.ToTensor(),
              # Normalizes from [0, 1] to [-1, 1] generically safely
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.samples = self._load_dataset_paths()

    def _load_dataset_paths(self):
        samples = []
        
        split_file = os.path.join(self.data_dir, f"{self.split}_split.json")
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found. Please run pipeline/data_preparation.py first.")
            return []
            
        with open(split_file, "r") as f:
            video_paths = json.load(f)
            
        for vid_path in video_paths:
            base_name = os.path.basename(vid_path).replace('.mp4', '')
            original_speaker_dir = os.path.dirname(os.path.dirname(vid_path))
            speaker_name = os.path.basename(original_speaker_dir)
            
            # Reconstruct paths using the current data_dir
            vid_path = os.path.join(self.data_dir, speaker_name, 'video', f"{base_name}.mp4")
            align_path = os.path.join(self.data_dir, speaker_name, 'align', f"{base_name}.align")
            audio_path = os.path.join(self.data_dir, speaker_name, 'audio', f"{base_name}.wav")
            
            if os.path.exists(align_path):
                transcript = self._read_align_file(align_path)
                
                # Check for audio if requested
                if self.use_audio and not os.path.exists(audio_path):
                    continue
                    
                if transcript:
                    sample_dict = {
                        "video": vid_path,
                        "transcript": transcript,
                        "align_path": align_path
                    }
                    if self.use_audio:
                        sample_dict["audio"] = audio_path
                        
                    samples.append(sample_dict)
                    
        return samples

    def _read_align_file(self, align_path):
        words = []
        with open(align_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, word = parts
                    if word not in ['sil', 'sp']:
                        words.append(word)
        return " ".join(words)

    def _load_video_frames(self, path):
        """ Loads already pre-cropped and pre-sized video """
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Keep true length so downstream modules can build padding masks.
        true_len = min(len(frames), self.max_frames)

        # Temporal padding just in case length doesn't strictly match max_frames
        if len(frames) == 0:
            frames = [torch.zeros((3, 96, 96)) for _ in range(self.max_frames)]
            return torch.stack(frames), 0
            
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        else:
            pad_len = self.max_frames - len(frames)
            zero_frame = frames[0] * 0  # zero-filled frame same shape as real frames
            frames.extend([zero_frame] * pad_len)
            
        tensor_list = []
        for f in frames:
            tf = self.transform(f) if self.transform else torch.tensor(f)
            if not isinstance(tf, torch.Tensor):
                tf = torch.tensor(tf)
            tensor_list.append(tf)
            
        return torch.stack(tensor_list), true_len

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path, backend="ffmpeg")
        # Resample if the file's sample rate doesn't match the expected rate
        if sr != _MEL_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, _MEL_SAMPLE_RATE)
        target_len = _MEL_SAMPLE_RATE * 3
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        # Compute log-mel here in the data-loading worker (CPU), not in the
        # GPU forward pass, so the model receives B x T_mel x n_mels tensors.
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=_MEL_SAMPLE_RATE,
            n_mels=_MEL_N_MELS,
            n_fft=_MEL_N_FFT,
            hop_length=_MEL_HOP_LENGTH,
        )
        mel_spec = mel_transform(waveform)          # 1 x n_mels x T_mel
        log_mel = torch.log(mel_spec + 1e-6).squeeze(0).T  # T_mel x n_mels
        return log_mel
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames_tensor, video_len = self._load_video_frames(sample["video"])
        
        ret_dict = {
            "video": frames_tensor,
            "transcript": sample["transcript"],
            "video_len": video_len
        }
        
        if self.use_audio and "audio" in sample:
            audio_tensor = self._load_audio(sample["audio"])
            ret_dict["audio"] = audio_tensor
            
        return ret_dict

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from torch.utils.data import DataLoader
    from utils.dataset_utils import Config
    from utils.phoneme_converter import PhonemeConverter
    from utils.negative_sampler import NegativeSampler

    config = Config()
    print("Initializing GRIDDataset in 'train' split...")
    # Adjust 'data_dir' to the actual preprocessed dataset path
    print(f"Looking for data in: {config.processed_data_dir}")
    dataset = GRIDDataset(
        data_dir=config.processed_data_dir,
        split="train",
        max_frames=config.max_frames,
        use_audio=config.use_audio
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        
        print("\n--- Sample Batch Info ---")
        print(f"Video batch shape (B, T, C, H, W): {batch['video'].shape}")
        print(f"Transcripts: {batch['transcript']}")
        if "audio" in batch:
            print(f"Audio batch shape (B, channels, samples): {batch['audio'].shape}")

        print("\n--- Positive/Negative Debug Samples ---")
        converter = PhonemeConverter()
        transcript_pool = [item["transcript"] for item in dataset.samples]
        sampler = NegativeSampler(transcript_pool)

        sample_count = min(3, len(dataset))
        for idx in range(sample_count):
            sample = dataset[idx]
            pos_transcript = sample["transcript"]
            pos_phonemes, pos_indices = converter.text_to_phonemes(pos_transcript)

            neg_transcript = sampler.sample(pos_transcript)
            neg_phonemes, neg_indices = converter.text_to_phonemes(neg_transcript)

            print(f"\nSample {idx + 1} Positive:")
            print(f"Transcript: '{pos_transcript}'")
            print(f"Phonemes: {pos_phonemes}")
            print(f"Phoneme indices: {pos_indices}")
            print(f"Video Tensor Shape: {sample['video'].shape}")
            print(f"Video Length (unpadded): {sample['video_len']}")
            if "audio" in sample:
                print(f"Audio Tensor Shape: {sample['audio'].shape}")

            print(f"Sample {idx + 1} Negative:")
            print(f"Transcript: '{neg_transcript}'")
            print(f"Phonemes: {neg_phonemes}")
            print(f"Phoneme indices: {neg_indices}")
    else:
        print("Dataset is empty. Ensure data is correctly preprocessed and paths are valid.")
