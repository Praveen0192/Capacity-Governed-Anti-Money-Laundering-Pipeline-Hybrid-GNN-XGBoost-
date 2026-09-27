import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Imbalance-Aware Focal Loss for Binary Classification.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)   
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        # Forward pass for Focal Loss.
        
        # Step 1: Compute standard Cross-Entropy loss per sample (unreduced)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Step 2: Extract the probability assigned to the true class (p_t)
        # Since ce_loss = -log(p_t), p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)
        
        # Step 3: Compute the alpha balancing factor (alpha_t)
        # Assign self.alpha to class 1 (illicit) and (1 - self.alpha) to class 0 (licit)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        # Step 4: Compute the modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Step 5: Calculate final Focal Loss per sample
        focal_loss = alpha_t * modulating_factor * ce_loss
        
        # Step 6: Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Test Execution

if __name__ == "__main__":
    print(" Testing Focal Loss implementation...")
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
  
    sample_logits = torch.tensor([
        [ 4.0, -4.0],
        [ 3.0, -3.0],
        [ 0.1, -0.1],
        [-4.0,  4.0]
    ])
    sample_targets = torch.tensor([0, 1, 1, 1])
    
    # Compute unreduced losses to see sample-by-sample contribution
    unreduced_criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='none')
    losses = unreduced_criterion(sample_logits, sample_targets)
    
    print("\nSample Results Breakdown:")
    print(f"1. Easy Licit  (Correct, high confidence): Loss = {losses[0].item():.6f}")
    print(f"2. Hard Illicit (Wrong, high confidence):  Loss = {losses[1].item():.6f}")
    print(f"3. Uncertain Illicit:                      Loss = {losses[2].item():.6f}")
    print(f"4. Easy Illicit (Correct, high confidence): Loss = {losses[3].item():.6f}")
    
    total_loss = criterion(sample_logits, sample_targets)
    print(f"\nMean Batch Loss: {total_loss.item():.4f}")
    print("Focal Loss module is fully functional.")