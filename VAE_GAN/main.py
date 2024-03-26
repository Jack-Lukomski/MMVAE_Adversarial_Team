import torch
from mmvae.models.HumanMouseVAE import SharedVA
from mmvae.trainers.MultiModel_gan import HumanMouseVAETrainer

def main():
    # Define training parameters
    batch_size = 64
    learning_rate = 0.001
    discriminator_learning_rate = 0.0001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = SharedVA().to(device)

    # Initialize the trainer
    trainer = HumanMouseVAETrainer(model=model,
                                   batch_size=batch_size,
                                   lr=learning_rate,
                                   discriminator_lr=discriminator_learning_rate,
                                   device=device)

    # Start training
    for epoch in range(1, num_epochs + 1):
        print(f'Starting Epoch {epoch}')
        trainer.train_epoch(epoch)

    print('Training Complete')

if __name__ == '__main__':
    main()
