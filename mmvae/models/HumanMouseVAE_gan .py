import torch
import torch.nn as nn
import torch.nn.functional as F
import mmvae.models as M
import mmvae.models.utils as utils

class SharedVAE(M.VAE):

    _initialized = None

    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear, init_weights = False):
        super(SharedVAE, self).__init__(encoder, decoder, mean, var)

        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True
        

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        
        utils._submodules_init_weights_xavier_uniform_(self.encoder)
        utils._submodules_init_weights_xavier_uniform_(self.decoder)
        utils._submodules_init_weights_xavier_uniform_(self.mean)
        utils._xavier_uniform_(self.var, -1.0) # TODO: Add declare why

class HumanExpert(M.Expert):

    _initialized = None

    def __init__(self, encoder, decoder, discriminator, init_weights = False):
        super(HumanExpert, self).__init__(encoder, decoder, discriminator)

        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        utils._submodules_init_weights_xavier_uniform_(self)

class MouseExpert(M.Expert):

    _initialized = None

    def __init__(self, encoder, decoder, discriminator, init_weights = False):
        super(HumanExpert, self).__init__(encoder, decoder, discriminator)

        if init_weights:
            print("Initialing SharedEncoder xavier uniform on all submodules")
            self.__init__weights()
        self._initialized = True

    def __init__weights(self):
        if self._initialized:
            raise RuntimeError("Cannot invoke after intialization!")
        utils._submodules_init_weights_xavier_uniform_(self)

class Model(nn.Module):
    def __init__(self, human_expert, mouse_expert, shared_vae):
        super(Model, self).__init__()
        self.human_expert = human_expert
        self.mouse_expert = mouse_expert
        self.shared_vae = shared_vae

    def forward(self, x, species):
        if species == 'human':
            expert = self.human_expert
        elif species == 'mouse':
            expert = self.mouse_expert
        else:
            raise ValueError("Unsupported species")
        
        x = expert.encoder(x)
        latent_representation, mu, var = self.shared_vae(x)
        species_pred = self.new_discriminator(latent_representation)
        reconstructed_x = expert.decoder(latent_representation)
        return reconstructed_x, mu, var, species_pred

class SharedEncoder(nn.Module):

    _initialized = None

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
        nn.Sequential(
            nn.Linear(60664, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
        nn.Sequential(
            nn.Linear(60664, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
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

    model = Model(human_expert, mouse_expert, shared_vae)

    return model

