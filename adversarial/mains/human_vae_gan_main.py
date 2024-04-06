import torch
import json
from adversarial.trainers import human_vae_gan

HPARAMS_PATH = '/active/debruinz_project/linhao_y/MMVAE_Adversarial_Team/adversarial/hparams/human_vae_gan.json'
CSV_PATH = '/active/debruinz_project/linhao_y/MMVAE_Adversarial_Team/adversarial/results/loss.csv'

def main(device):

    with open(HPARAMS_PATH, mode='r') as file:
        json_hparams = json.load(file)

    hparams = human_vae_gan.HumanVAEGANConfig(config=json_hparams)
    trainer = human_vae_gan.HumanVAEGANTrainer(device=device, hparams=hparams)

    trainer.train(hparams['epochs'])

    trainer.get_run_csv(CSV_PATH)

if __name__ == "__main__":
    CUDA = True
    device = "cuda" if torch.cuda.is_available() and CUDA else "cpu"
    main(device)
