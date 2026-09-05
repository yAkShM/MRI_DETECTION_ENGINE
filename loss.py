import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)


        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        #intersection:
        intersection = torch.sum(probs*targets, dim=2)
        #cardinality
        cardinality = torch.sum(probs**2, dim=2)+ torch.sum(targets**2, dim=2)

        dice_score = (2*intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = 1- dice_score

        return dice_loss.mean()

class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss(smooth = smooth)
        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, logits, targets):

        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets)

        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)



if __name__ == "__main__":
    # --- Sanity Check ---
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # 1. Create dummy predictions with gradient tracking: [B=1, C=3, D=16, H=32, W=32]
    # (Using smaller spatial dims for a quick local CPU check)
    dummy_logits = torch.randn(1, 3, 16, 32, 32, requires_grad=True)
    
    # 2. Create dummy binary targets (0 or 1)
    dummy_targets = torch.randint(0, 2, (1, 3, 16, 32, 32)).float()

    # 3. Compute loss
    loss = criterion(dummy_logits, dummy_targets)
    print(f"Calculated Combined Loss: {loss.item():.4f}")

    # 4. Verify backpropagation
    loss.backward()
    print("Backward pass verified: Gradients computed successfully.")
    print(f"Gradient Tensor Shape on Logits: {dummy_logits.grad.shape}")
