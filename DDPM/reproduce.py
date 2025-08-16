import torch
import numpy as np
from ddpm import DDPM, NoiseScheduler

@torch.no_grad()
def sample(model, x_T, noise_scheduler):
    """
    Perform deterministic sampling using provided initial noise x_T and setting z=0 at all steps.
    
    Args:
        model: Trained DDPM model
        x_T: Initial noise tensor [n_samples, n_dim]
        noise_scheduler: Noise scheduler used during training
        
    Returns:
        Generated samples tensor [n_samples, n_dim]
    """
    model.eval()
    device = next(model.parameters()).device

    # Precompute scheduler parameters
    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    x_t = x_T.to(device)

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((x_t.size(0),), t, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_tensor)

        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute mean using predicted noise
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) / sqrt_alpha_t
        
        # Set z to zero for deterministic sampling
        z = torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(beta_t) * z

        x_t = x_prev

    return x_t

def main():
    n_dim = 64
    n_steps = 200
    lbeta=0.00001
    ubeta=0.02
    model_path = f'exps/ddpm_albatross_{n_dim}_{n_steps}_{lbeta}_{ubeta}/model.pth'
    prior_samples_path = "data/albatross_prior_samples.npy"
    output_path = "albatross_samples.npy"

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prior samples
    prior_samples = np.load(prior_samples_path)
    prior_tensor = torch.from_numpy(prior_samples).float().to(device)

    # Initialize model and noise scheduler
    model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
    noise_scheduler = NoiseScheduler(
        num_timesteps=n_steps,
        type="linear",
        beta_start=lbeta,
        beta_end=ubeta,
    )

    # Load trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Generate samples deterministically
    samples = sample(model, prior_tensor, noise_scheduler)

    # Save generated samples
    samples_np = samples.cpu().numpy()
    np.save(output_path, samples_np)
    print(f"Samples saved to {output_path}")

if __name__ == "__main__":
    main()