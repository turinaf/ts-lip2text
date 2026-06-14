import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    all_scores, all_aligns, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            phonemes = batch['phonemes'].to(device)
            labels = batch['label'].to(device)

            audio = batch.get('audio')
            if audio is not None:
                audio = audio.to(device)

            phoneme_padding_mask = batch.get('phoneme_padding_mask')
            if phoneme_padding_mask is not None:
                phoneme_padding_mask = phoneme_padding_mask.to(device)

            video_padding_mask = batch.get('video_padding_mask')
            if video_padding_mask is not None:
                video_padding_mask = video_padding_mask.to(device)

            consistency_score, align_out = model(
                video,
                phonemes,
                audio,
                video_padding_mask=video_padding_mask,
                phoneme_padding_mask=phoneme_padding_mask
            )
            
            all_scores.append(consistency_score.view(-1))
            all_aligns.append(align_out["alignment_score"])
            all_labels.append(labels)
            
    all_scores = torch.cat(all_scores)
    all_aligns = torch.cat(all_aligns)
    all_labels = torch.cat(all_labels)
    
    from evaluation.metrics import compute_metrics
    metrics = compute_metrics(all_labels, all_scores, all_aligns)
    
    print(f"ROC AUC: {metrics['roc_auc']:.4f}, Verify Acc: {metrics['verification_accuracy']:.4f}, Align Acc: {metrics['alignment_accuracy']:.4f}")
    return metrics
