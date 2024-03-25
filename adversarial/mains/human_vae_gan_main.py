from mmvae.trainers import HumanVAE_gan
import torch

def main(device):
    # Define any hyperparameters
    batch_size = 32
    # Create trainer instance
    trainer = HumanVAE_gan.HumanVAETrainer(
        batch_size=batch_size,
        device=device,
        log_dir="/active/debruinz_project/jack_lukomski/03_23_2024_logs/baseline_test",
        lr=0.001,
        annealing_steps=1,
        discriminator_div=1
    )
    # Train model with number of epochs
    trainer.train(epochs=5)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)