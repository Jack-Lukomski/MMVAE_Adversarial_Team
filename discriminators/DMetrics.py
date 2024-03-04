import torch
from sklearn.metrics import roc_curve, auc as calc_roc
import numpy as np

class DMetrics:
    def __init__(self, generator, dataloader):
        self.generator = generator
        self.dataloader = dataloader

    def get_roc(self, discriminator):
      real_scores = []
      fake_scores = []

      with torch.no_grad():
        for train_data in self.dataloader:
          fake_data = self.generator(train_data)
          real_scores.extend(discriminator(train_data).view(-1).tolist())
          fake_scores.extend(discriminator(fake_data).view(-1).tolist())

      scores = np.array(real_scores + fake_scores)
      y_true = np.array([1] * len(real_scores) + [0] * len(fake_scores))
      fpr, tpr, thresholds = roc_curve(y_true, scores)
      roc_auc = calc_roc(fpr, tpr)

      return fpr, tpr, roc_auc
    
    def md_eval(self, md, md_epochs):
      md_optimizer = torch.optim.Adam(md.parameters(), lr=0.0001)
      md_loss_fn = torch.nn.MSELoss()

      for epoch in range(md_epochs):
        for train_data in self.dataloader:
          md_optimizer.zero_grad()

          real_pred = md(train_data)
          real_loss = md_loss_fn(real_pred, torch.ones_like(real_pred))
          real_loss.backward()

          fake_data = self.generator(train_data)

          fake_pred = md(fake_data)
          fake_loss = md_loss_fn(fake_pred, torch.zeros_like(fake_pred))
          fake_loss.backward()

          md_optimizer.step()

      return self.get_roc(md)