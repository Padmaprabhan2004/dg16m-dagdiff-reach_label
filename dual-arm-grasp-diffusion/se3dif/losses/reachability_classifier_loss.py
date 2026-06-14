import torch
import torch.nn as nn
from se3dif.utils import SO3_R3


class ReachabilityClassifierLoss():
    def __init__(self, field='reachability_classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCELoss()

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        gt_labels = ground_truth['reach_labels']
        
        # Compute classifier loss
        if model.reachability_classifier is not None:
            reach_classifier_loss = self.bce_loss(model.reach_label.reshape(-1), gt_labels.reshape(-1))/5
        else:
            reach_classifier_loss = 0
        
        acc = torch.sum((model.reach_label.reshape(-1) > 0.5).float() == gt_labels.reshape(-1)).item() / gt_labels.numel()

        info = {self.field: model.reach_label.reshape(-1)}
        loss_dict = {"Classifier Loss": reach_classifier_loss,
                     'Classifier Accuracy': acc}
        
        return loss_dict, info
