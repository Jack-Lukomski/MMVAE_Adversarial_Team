import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M
import mmvae.models.utils as utils

class SharedVAE(M.VAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear, init_weights=False):
        super(SharedVAE, self).__init__(encoder, decoder, mean, var)
        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after initialization!")
        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.mean)
        utils._xavier_uniform_(self.var, -1.0)

class HumanExpert(M.Expert):
    def __init__(self, encoder, decoder, init_weights=False):
        super(HumanExpert, self).__init__(encoder, decoder)
        if init_weights:
            print("Initializing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()

    def __init__weights(self):
        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)

class MouseExpert(M.Expert):
    def __init__(self, encoder, decoder, init_weights=False):
        super(MouseExpert, self).__init__(encoder, decoder)
        if init_weights:
            print("Initializing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()

    def __init__weights(self):
        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SharedDecoder(nn.Module):
    def __init__(self):
        super(SharedDecoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 768)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x

class Model(nn.Module):
    def __init__(self, human_expert, mouse_expert, shared_vae, VAE_discriminator):
        super(Model, self).__init__()
        self.human_expert = human_expert
        self.mouse_expert = mouse_expert
        self.shared_vae = shared_vae
        self.VAE_discriminator = VAE_discriminator

    def forward(self, x, species):
        if species == 'human':
            expert = self.human_expert
        elif species == 'mouse':
            expert = self.mouse_expert
        else:
            raise ValueError("Unsupported species")

        x = expert.encoder(x)
        latent_representation, mu, var = self.shared_vae(x)
        reconstructed_x = expert.decoder(latent_representation)
        species_pred = self.VAE_discriminator(latent_representation)
        return reconstructed_x, mu, var, species_pred

def configure_model() -> Model:
    human_expert = HumanExpert(
        nn.Sequential(
            nn.Linear(60664, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 60664),
            nn.LeakyReLU()
        ),
        init_weights=True
    )

    mouse_expert = MouseExpert(
        nn.Sequential(
            nn.Linear(60664, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
        ),
        nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 60664),
            nn.LeakyReLU()
        ),
        init_weights=True
    )

    shared_vae = SharedVAE(
        SharedEncoder(),
        SharedDecoder(),
        nn.Linear(256, 128),
        nn.Linear(256, 128),
        init_weights=True
    )

    VAE_discriminator = nn.Sequential(
        nn.Linear(128, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    model = Model(human_expert, mouse_expert, shared_vae, VAE_discriminator)

    return model