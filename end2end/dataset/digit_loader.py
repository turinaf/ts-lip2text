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

class DigitDataset(Dataset):
    def __init__(self, data_dir, split='train', max_frames=75, transform=None, use_audio=False):
        """
        Loads the preprocessed Digit dataset using pre-computed split files.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.use_audio = use_audio
        self.transform = transform or T.Compose([
            T.ToTensor(),
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
            
        for vid_path_raw in video_paths:
            base_name = os.path.basename(vid_path_raw).replace('.mp4', '')
            # original_speaker_dir = "video"
            # os.path.dirname(original_speaker_dir) = "1011"
            speaker_dir_path = os.path.dirname(os.path.dirname(vid_path_raw))
            speaker_name = os.path.basename(speaker_dir_path)
            subset_name = os.path.basename(os.path.dirname(speaker_dir_path))
            
            # The digit dataset uses 'text' directory and .txt files
            # Reconstruct paths using the current data_dir
            vid_path = os.path.join(self.data_dir, subset_name, speaker_name, 'video', f"{base_name}.mp4")
            align_path = os.path.join(self.data_dir, subset_name, speaker_name, 'text', f"{base_name}.txt")
            audio_path = os.path.join(self.data_dir, subset_name, speaker_name, 'audio', f"{base_name}.wav")
            
            # The paths in the JSON already contain the data_dir prefix! So we don't need to rebuild them if they are absolute/relative from root
            if "data/digit" in vid_path_raw: # The original passed in vid_path
                vid_path = vid_path_raw.replace('./', '')
                align_path = vid_path.replace('/video/', '/text/').replace('.mp4', '.txt')
                audio_path = vid_path.replace('/video/', '/audio/').replace('.mp4', '.wav')
                
            # Allow searching since exlamation points (!) might be escaped/missing and names might vary slightly in the actual filesystem
            if not os.path.exists(vid_path):
                 # Try matching without escaping/special characters if exact match fails
                 potential_matches = glob.glob(os.path.join(self.data_dir, subset_name, speaker_name, 'video', f"*{base_name.split('_')[-2]}*.mp4"))
                 if potential_matches:
                     vid_path = potential_matches[0]
                     base_name = os.path.basename(vid_path).replace('.mp4', '')
                     align_path = os.path.join(self.data_dir, subset_name, speaker_name, 'text', f"{base_name}.txt")
                     audio_path = os.path.join(self.data_dir, subset_name, speaker_name, 'audio', f"{base_name}.wav")
                     
            if os.path.exists(align_path):
                transcript = self._read_text_file(align_path)
                
                # Check for audio if requested
                if self.use_audio and not os.path.exists(audio_path):
                    print(f"Missing audio: {audio_path}")
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
            else:
                print(f"Missing: align={align_path}, audio={audio_path}, video={vid_path}")
                break
                
        return samples

    def _read_text_file(self, align_path):
        # The digit text file just contains digits e.g. "4 5 6 3 8 7 1 6"
        with open(align_path, 'r') as f:
            content = f.read().strip()
        return content

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        true_len = min(len(frames), self.max_frames)

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
        # Compute log-mel in the data-loading worker (CPU), not in the GPU forward pass
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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.dataset_utils import Config
    from utils.negative_sampler import NegativeSampler
    
    config = Config()
    test_data_dir = config.processed_data_dir
    
    print(f"--- Debugging DigitDataset ---")
    print(f"Looking for data in: {test_data_dir}")
    print(f"Using audio: {config.use_audio}")
    
    try:
        dataset = DigitDataset(data_dir=test_data_dir, split='train', max_frames=config.max_frames, use_audio=config.use_audio)
        print(f"\nSuccessfully loaded train split with {len(dataset)} samples.")

        digit_vocab = {str(i): i + 1 for i in range(10)}
        digit_vocab["!"] = 11
        digit_vocab["<pad>"] = 0
        digit_vocab["<unk>"] = 12

        def transcript_to_sequence(transcript):
            return [digit_vocab.get(token, digit_vocab["<unk>"]) for token in transcript.split()]
        
        if len(dataset) > 0:
            print("\n--- Inspecting Samples ---")
            transcript_pool = [sample_item["transcript"] for sample_item in dataset.samples]
            sampler = NegativeSampler(transcript_pool)

            sample_count = min(3, len(dataset))
            for idx in range(sample_count):
                sample = dataset[idx]
                positive_seq = transcript_to_sequence(sample["transcript"])
                negative_transcript = sampler.sample(sample["transcript"])
                negative_seq = transcript_to_sequence(negative_transcript)

                print(f"\nSample {idx + 1} Positive:")
                print(f"Transcript: '{sample['transcript']}'")
                print(f"Sequence: {positive_seq}")
                print(f"Video Tensor Shape: {sample['video'].shape}")
                print(f"Video Tensor Data Type: {sample['video'].dtype}")
                print(f"Max Val: {sample['video'].max():.4f}, Min Val: {sample['video'].min():.4f}")
                if "audio" in sample:
                    print(f"Audio Tensor Shape: {sample['audio'].shape}")

                print(f"Sample {idx + 1} Negative:")
                print(f"Transcript: '{negative_transcript}'")
                print(f"Sequence: {negative_seq}")
            
    except Exception as e:
        print(f"\nError initializing dataset: {e}")
