import os
import json
import torch
from utils.phoneme_converter import PhonemeConverter
from utils.negative_sampler import NegativeSampler

class Config:
    def __init__(self, config_path="config.json"):
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {}
        self.dataset = cfg.get("dataset", "grid")
        self.processed_data_dir = cfg.get("digit_processed_data_dir") if self.dataset == "digit" else cfg.get("grid_processed_data_dir")
        self.use_audio = cfg.get("use_audio", False)
        self.batch_size = cfg.get("batch_size", 32)
        self.learning_rate = cfg.get("learning_rate", 0.0001)
        self.epochs = cfg.get("epochs", 100)
        self.max_frames = cfg.get("max_frames", 75)
        self.output_dir = cfg.get("output_dir", "./exp/no_audio")
        self.lip_backbone_pretrained = cfg.get("lip_backbone_pretrained", True)

def collate_fn_factory(dataset_name, transcript_pool=None):
    converter = PhonemeConverter() if dataset_name != "digit" else None
    global_sampler = NegativeSampler(transcript_pool or [])
    
    digit_vocab = {str(i): i+1 for i in range(10)} 
    digit_vocab["!"] = 11
    digit_vocab["<pad>"] = 0
    digit_vocab["<unk>"] = 12

    def collate_fn(batch):
        all_trans = [item['transcript'] for item in batch]
        sampler = global_sampler if len(global_sampler.all_transcripts) > 0 else NegativeSampler(all_trans)
        
        videos, phonemes_indices_list, labels, transcripts, audios, video_lens = [], [], [], [], [], []
        use_audio = 'audio' in batch[0]
        
        for item in batch:
            vid, trans = item['video'], item['transcript']
            if use_audio: aud = item['audio']
            
            # --- POSITIVE SAMPLE ---
            videos.append(vid)
            if dataset_name == "digit":
                pos_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in trans.split()]
            else:
                _, pos_indices = converter.text_to_phonemes(trans)
            phonemes_indices_list.append(torch.tensor(pos_indices, dtype=torch.long))
            
            labels.append(1.0)
            transcripts.append(trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio: audios.append(aud)
            
            # --- NEGATIVE SAMPLE ---
            videos.append(vid)
            neg_trans = sampler.sample(trans)
            if dataset_name == "digit":
                neg_indices = [digit_vocab.get(d, digit_vocab["<unk>"]) for d in neg_trans.split()]
            else:
                _, neg_indices = converter.text_to_phonemes(neg_trans)
            phonemes_indices_list.append(torch.tensor(neg_indices, dtype=torch.long))
            
            labels.append(0.0)
            transcripts.append(neg_trans)
            video_lens.append(int(item.get('video_len', vid.shape[0])))
            if use_audio: audios.append(aud)
            
        videos = torch.stack(videos)
        labels = torch.tensor(labels, dtype=torch.float32)
        if use_audio: audios = torch.stack(audios)
            
        padded_phonemes = torch.nn.utils.rnn.pad_sequence(phonemes_indices_list, batch_first=True, padding_value=0)
        phoneme_padding_mask = padded_phonemes.eq(0)

        max_t = videos.shape[1]
        video_lens_t = torch.tensor(video_lens, dtype=torch.long)
        video_padding_mask = torch.arange(max_t).unsqueeze(0) >= video_lens_t.unsqueeze(1)
            
        batch_dict = {
            'video': videos,
            'phonemes': padded_phonemes,
            'phoneme_padding_mask': phoneme_padding_mask,
            'video_padding_mask': video_padding_mask,
            'label': labels,
            'transcript': transcripts
        }
        if use_audio: batch_dict['audio'] = audios
            
        return batch_dict
    return collate_fn

# Default instances for tests
default_config = Config()
collate_fn = collate_fn_factory(default_config.dataset)
