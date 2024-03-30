import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve, auc

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction="sum"):
    """
    Calculate the KL divergence between a given Gaussian distribution q(z|x)
    and the standard Gaussian distribution p(z).

    Parameters:
    - mu (torch.Tensor): The mean of the Gaussian distribution q(z|x).
    - sigma (torch.Tensor): The standard deviation of the Gaussian distribution q(z|x).
    - beta (int): Default = 0.5 - Weight in which to factor KL Divergence. 

    Returns:
    - torch.Tensor: The KL divergence.
    """
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "mean":
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

def cyclic_annealing(batch_iteration, cycle_length, min_beta=0.0, max_beta=1.0, ceil_downswings=True, floor_upswings=False):
    """
    Calculates the cyclic annealing rate based on the current batch iteration.
    
    Parameters:
    - batch_iteration: Current batch iteration in the training process.
    - cycle_length: Number of batch iterations in a full cycle.
    - min_beta: Minimum value of the annealing rate.
    - max_beta: Maximum value of the annealing rate.
    - ceil_upswings: Keeps downswings at max_beta.
    - floor_upswings: Keeps upswings at min_beta.
    
    Returns:
    - beta_value: The calculated annealing rate for the current batch iteration.
    """
    
    if (cycle_length <= 0):
        raise RuntimeError("Cycle length must be greater than 0!")

    # Determine the current position in the cycle
    cycle_position = batch_iteration % cycle_length
    # Calculate the phase of the cycle (upswing or downswing)
    if cycle_position < cycle_length // 2:
        if floor_upswings:
            return min_beta
        # Upswing phase
        return min_beta + (max_beta - min_beta) * (2 * cycle_position / cycle_length)
    else:
        if ceil_downswings:
            return max_beta
        # Downswing phase
        return max_beta - (max_beta - min_beta) * (2 * (cycle_position - cycle_length / 2) / cycle_length)

def calculate_r2(input: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates the R^2 score indicating how well the target reconstructs the input.

    Parameters:
    - input: torch.Tensor representing the original data.
    - target: torch.Tensor representing the reconstructed or predicted data.

    Returns:
    - R^2 score as a float.
    """
    # Ensure input and target have the same shape
    if input.shape != target.shape:
        raise ValueError("Input and target tensors must have the same shape")

    # Calculate the mean of the original inputs
    mean_input = input.mean()

    # Calculate SS_tot (total sum of squares of difference from the mean)
    ss_tot = torch.sum((input - mean_input) ** 2)

    # Calculate SS_res (sum of squares of the residuals between input and target)
    ss_res = torch.sum((input - target) ** 2)

    # Calculate and return the R^2 score
    r2_score = 1 - ss_res / ss_tot
    return r2_score.item()
    
def build_non_zero_mask(crow_indices, col_indices, shape):
    """
    Build a mask for non-zero elements in a CSR-like sparse matrix.
    
    Parameters:
    - crow_indices: Compressed row indices from crow_indices() method.
    - col_indices: Column indices for each non-zero element from col_indices() method.
    - shape: The shape of the full matrix (rows, cols).
    
    Returns:
    - A 2D list (or any suitable structure) where True represents a non-zero element,
      and False represents a zero element.
    """
    rows, cols = shape
    mask = [[False] * cols for _ in range(rows)]  # Initialize mask with all False (zero elements)
    
    for row in range(rows):
        start_pos = crow_indices[row]
        end_pos = crow_indices[row + 1]
        for idx in range(start_pos, end_pos):
            col = col_indices[idx]
            mask[row][col] = True  # Mark non-zero positions as True
    
    return mask

def batch_roc(bc, real_batch_data, fake_batch_data):
    real_scores = bc(real_batch_data).view(-1).cpu().detach().numpy()
    fake_scores = bc(fake_batch_data).view(-1).cpu().detach().numpy()

    scores = np.concatenate([real_scores, fake_scores])
    y_true = np.concatenate([np.ones_like(real_scores), np.zeros_like(fake_scores)])
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def md_eval(md, md_epochs, gen, dataloader):
  md_optimizer = torch.optim.Adam(md.parameters(), lr=1e-6)
  md_loss_fn = torch.nn.BCELoss()

  for epoch in range(md_epochs):
    for (train_data, label) in dataloader:
      md_optimizer.zero_grad()

      real_pred = md(train_data)
      if torch.isnan(real_pred).any():
          print("NaN in D output for real data")
      real_loss = md_loss_fn(real_pred, torch.ones_like(real_pred))
      if torch.isnan(real_loss).any():
          print("NaN in real_loss")
      real_loss.backward()

      fake_data = gen(train_data)[0]
      if torch.isnan(fake_data).any():
          print("NaN in G output")

      fake_pred = md(fake_data)
      if torch.isnan(fake_pred).any():
          print("NaN in D output for fake data")
      fake_loss = md_loss_fn(fake_pred, torch.zeros_like(fake_pred))
      if torch.isnan(fake_loss).any():
          print("NaN found in fake_loss")
      fake_loss.backward()


      for name, param in md.named_parameters():
          if param.grad is not None and torch.isnan(param.grad).any():
              print(f'NaN gradient detected in {name}')

      md_optimizer.step()

    real_scores = []
    fake_scores = []

    with torch.no_grad():
        for (train_data, label) in dataloader:
            fake_data = gen(train_data)[0]
            real_scores.extend(md(train_data).view(-1).tolist())
            fake_scores.extend(md(fake_data).view(-1).tolist())

    scores = np.array(real_scores + fake_scores)
    y_true = np.array([1] * len(real_scores) + [0] * len(fake_scores))
    if np.isnan(scores).any():
        print("NaN found in scores array")
    if np.isnan(y_true).any():
        print("NaN found in y_true")

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

class BatchPCC:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_x = 0
        self.sum_y = 0
        self.sum_x2 = 0
        self.sum_y2 = 0
        self.sum_xy = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        self.sum_x += y_true.sum().item()
        self.sum_y += y_pred.sum().item()
        self.sum_x2 += (y_true ** 2).sum().item()
        self.sum_y2 += (y_pred ** 2).sum().item()
        self.sum_xy += (y_true * y_pred).sum().item()
        self.n += y_true.size(0)

    def compute(self):
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        covariance = (self.sum_xy / self.n) - (mean_x * mean_y)
        variance_x = (self.sum_x2 / self.n) - (mean_x ** 2)
        variance_y = (self.sum_y2 / self.n) - (mean_y ** 2)
        pcc = covariance / torch.sqrt(torch.tensor(variance_x * variance_y))
        return pcc
