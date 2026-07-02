import torch
import torch.nn as nn
from se3dif.utils import SO3_R3


class ReachabilityClassifierLoss():
    def __init__(self, field='reachability_classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCEWithLogitsLoss()

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        gt = ground_truth['reach_labels'].float().reshape(-1)
        device = gt.device

        num_pos = gt.sum()
        num_neg = gt.numel() - num_pos
        # If the batch is imbalanced, upweight positive examples.
        # Clamp keeps the loss well-defined when a batch contains only one class.
        pos_weight = (num_neg / (num_pos + eps)).clamp(min=1.0, max=1e6)
        bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Compute classifier loss
        if model.reachability_classifier is not None:
            logits = model.reach_label.reshape(-1)
            reach_classifier_loss = bce_logits_loss(logits, gt)
        else:
            logits = torch.zeros_like(gt, device=device)
            reach_classifier_loss = torch.zeros((), device=device)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        tp = ((preds == 1) & (gt == 1)).sum().float()
        tn = ((preds == 0) & (gt == 0)).sum().float()
        fp = ((preds == 1) & (gt == 0)).sum().float()
        fn = ((preds == 0) & (gt == 1)).sum().float()

        recall_pos = tp / (tp+fn+eps)
        recall_neg = tn / (tn+fp+eps)
        balanced_acc=0.5*(recall_pos+recall_neg)

        precision =tp/(tp + fp + eps)
        f1 = 2.0*precision*recall_pos/(precision+recall_pos+eps)

        info = {self.field: probs}
        loss_dict = {"Reachability Classifier Loss": reach_classifier_loss,
                     'Reachability Accuracy': balanced_acc.item(),
                     'Reachability F1': f1.item()}
        
        return loss_dict, info

