from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def compute_metrics(labels, consistency_scores, align_scores):
    labels_np = labels.cpu().numpy()
    cons_np = consistency_scores.cpu().numpy()
    align_np = align_scores.cpu().numpy()
    
    # Decisions (verify if >= 0.5)
    preds = (cons_np >= 0.5).astype(int)
    align_preds = (align_np >= 0.5).astype(int) # simplistic align decision boundary
    
    auc = roc_auc_score(labels_np, cons_np) if len(set(labels_np)) > 1 else 0.0
    acc = accuracy_score(labels_np, preds)
    align_acc = accuracy_score(labels_np, align_preds)
    
    return {
        "roc_auc": auc,
        "verification_accuracy": acc,
        "alignment_accuracy": align_acc,
        "score_margin": np.mean(np.abs(cons_np[labels_np == 1] - cons_np[labels_np == 0])) if len(set(labels_np)) > 1 else 0.0
    }