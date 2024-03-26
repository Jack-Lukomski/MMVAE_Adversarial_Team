import torch
import torch.nn.functional as F
from torch.optim import Adam
import mmvae.models.HumanMouseVAE as HumanMouseVAE
from mmvae.trainers.trainer import BaseTrainer
from mmvae.data import MappedCellCensusDataLoader

class HumanMouseVAETrainer(BaseTrainer):
    """
    Trainer class for the HumanMouseVAE model, handling species-specific data.
    """
    def __init__(self, model: HumanMouseVAE.Model, batch_size: int, lr: float, discriminator_lr: float, device: torch.device):
        super(HumanMouseVAETrainer, self).__init__()
        self.model = model.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.discriminator_lr = discriminator_lr
        self.device = device
        # Initialize dataloaders for human and mouse data
        self.dataloader_human, self.dataloader_mouse = self.configure_dataloaders()
        # Initialize optimizers
        self.optimizers = self.configure_optimizers()

    def configure_dataloaders(self):
        human_data_path = '/active/debruinz_project/CellCensus_3M/3m_human_chunk_10.npz'
        mouse_data_path = '/active/debruinz_project/CellCensus_3M/3m_mouse_chunk_10.npz'
        dataloader_human = MappedCellCensusDataLoader(batch_size=self.batch_size, device=self.device, file_path=human_data_path, load_all=True)
        dataloader_mouse = MappedCellCensusDataLoader(batch_size=self.batch_size, device=self.device, file_path=mouse_data_path, load_all=True)
        return dataloader_human, dataloader_mouse

    def configure_optimizers(self):
        optimizers = {
            'shared_vae': Adam(self.model.shared_vae.parameters(), lr=self.lr),
            'human_discrim': Adam(self.model.human_expert.discriminator.parameters(), lr=self.discriminator_lr),
            'mouse_discrim': Adam(self.model.mouse_expert.discriminator.parameters(), lr=self.discriminator_lr),
        }
        return optimizers

    def train_epoch(self, epoch):
        self.model.train()
        # Iterate over human and mouse dataloaders
        for species, dataloader in [('human', self.dataloader_human), ('mouse', self.dataloader_mouse)]:
            for data in dataloader:
                self.train_batch(data, species)

    def train_batch(self, data, species):
        data = data.to(self.device)
        # Forward pass
        reconstructed_x, mu, var, species_pred = self.model(data, species)
        # Compute losses
        recon_loss = F.mse_loss(reconstructed_x, data)
        kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        species_label = torch.ones_like(species_pred, device=self.device) if species == 'human' else torch.zeros_like(species_pred, device=self.device)
        discrim_loss = F.binary_cross_entropy(species_pred, species_label)
        # Total loss for the VAE
        total_vae_loss = recon_loss + kl_loss
        # Backward and optimize
        self.optimizers['shared_vae'].zero_grad()
        total_vae_loss.backward(retain_graph=True)
        self.optimizers['shared_vae'].step()
        # Discriminator optimization
        discrim_optimizer_key = 'human_discrim' if species == 'human' else 'mouse_discrim'
        self.optimizers[discrim_optimizer_key].zero_grad()
        discrim_loss.backward()
        self.optimizers[discrim_optimizer_key].step()
