import torch

def dice_score(preds, targets, threshold=0.5, epsilon=1e-5):

    if preds.min()<0.0 or preds.max()>1.0:

        preds = torch.sigmoid(preds)


    preds_bin = (preds>threshold).float()
    targets = targets.float()

    preds_flat = preds_bin.view(preds_bin.size(0), preds_bin.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)

    intersection = torch.sum(preds_flat * targets_flat, dim=2)
    cardinality = torch.sum(preds_flat, dim=2)+torch.sum(targets_flat, dim=2)

    dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)

    return dice.mean(dim=0)

class MetricTracker:

    def __init__(self):

        self.reset()

    def reset(self):

        self.wt_scores = []
        self.tc_scores = []
        self.et_scores = []


    def update(self, preds, targets):

        with torch.no_grad():

            dice_per_channel = dice_score(preds, targets)

            self.wt_scores.append(dice_per_channel[0].item())
            self.tc_scores.append(dice_per_channel[1].item())
            self.et_scores.append(dice_per_channel[2].item())


    def get_results(self):

        mean_wt = sum(self.wt_scores)/ len(self.wt_scores) if self.wt_scores else 0.0
        mean_tc = sum(self.tc_scores)/ len(self.tc_scores) if self.tc_scores else 0.0
        mean_et = sum(self.et_scores)/ len(self.et_scores) if self.et_scores else 0.0

        overall_mean = (mean_et+mean_tc+mean_wt)/3.0

        return {
            "Dice_WT": mean_wt,
            "Dice_TC": mean_tc,
            "Dice_ET": mean_et,
            "Mean_Dice": overall_mean
        }


if __name__ == "__main__":
    # --- Sanity Check ---
    tracker = MetricTracker()

    # Simulate 2 validation batches of [B=1, C=3, D=16, H=32, W=32]
    for _ in range(2):
        dummy_preds = torch.randn(1, 3, 16, 32, 32)
        dummy_targets = torch.randint(0, 2, (1, 3, 16, 32, 32)).float()
        tracker.update(dummy_preds, dummy_targets)

    results = tracker.get_results()
    print("Metrics Sanity Check Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")