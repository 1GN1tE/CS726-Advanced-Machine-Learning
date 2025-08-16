import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import math

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="sigmoid", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def init_cosine_schedule(self, s=0.008):
        timesteps = self.num_timesteps
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start near 1
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0, 0.999)
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def init_sigmoid_schedule(self, start=-3, end=3, tau=3.0, clip_min=1e-9, clip_max=0.999):
        T = self.num_timesteps
        t = torch.linspace(0, 1, T)
        
        v_start = 1.0 / (1.0 + torch.exp(torch.tensor(-start / tau, dtype=torch.float32)))
        v_end   = 1.0 / (1.0 + torch.exp(torch.tensor(-end / tau,   dtype=torch.float32)))
        output = 1.0 / (1.0 + torch.exp(-((t * (end - start) + start) / tau)))
        gamma_t = (v_end - output) / (v_end - v_start)
        gamma_t = torch.clamp(gamma_t, clip_min, clip_max)
        
        self.alpha_bars = gamma_t.clone()
        alphas = torch.ones_like(self.alpha_bars)
        for i in range(1, T):
            denominator = max(self.alpha_bars[i-1].item(), clip_min)
            alphas[i] = torch.clamp(self.alpha_bars[i] / denominator, clip_min, 1.0)
        self.alphas = alphas
        self.betas = 1.0 - self.alphas


    def __len__(self):
        return self.num_timesteps

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps

        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Noise prediction model -> Implements ϵ_θ(x_t, t)
        self.model = nn.Sequential(
            nn.Linear(n_dim + 256, 512),
            nn.LayerNorm(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, n_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t_embed = self.time_embed(t)
        model_input = torch.cat([x, t_embed], dim=1)
        return self.model(model_input)

class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes=2, n_dim=3, n_steps=200):
        """
        Class dependernt noise prediction network for the DDPM

        Args:
            n_classes: number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.n_steps = n_steps

        # Time embedding layers
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Class embedding layers (including null class)
        self.class_embed = nn.Sequential(
            nn.Embedding(n_classes + 1, 256),  # +1 for null class
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Noise prediction model
        self.model = nn.Sequential(
            nn.Linear(n_dim + 512, 512),  # Concatenate x (n_dim) + time_embed (256) + class_embed (256)
            nn.LayerNorm(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, n_dim)
        )

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]
        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t_embed = self.time_embed(t)
        y_embed = self.class_embed(y)
        combined_embed = torch.cat([t_embed, y_embed], dim=1)
        model_input = torch.cat([x, combined_embed], dim=1)
        return self.model(model_input)
    
class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.T = noise_scheduler.num_timesteps
        self.n_classes = model.n_classes
        self.num_trials = 10

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """
        proba = self.predict_proba(x)
        return proba.argmax(dim=1)

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """
        device = x.device
        B = x.size(0)
        C = self.n_classes
        mse_accum = torch.zeros(B, C, device=device)

        for _ in range(self.num_trials):
            # Random timestep for each sample
            t = torch.randint(0, self.T, (B,), device=device).long()
            
            # Forward diffusion process
            alpha_bars = self.noise_scheduler.alpha_bars.to(device)
            alpha_bar_t = alpha_bars[t]
            noise = torch.randn_like(x)
            x_t = torch.sqrt(alpha_bar_t[:, None]) * x + torch.sqrt(1 - alpha_bar_t[:, None]) * noise

            # Predict noise for all classes
            x_t_rep = x_t.repeat_interleave(C, dim=0)  # (B*C, n_dim)
            t_rep = t.repeat_interleave(C, dim=0)
            y_labels = torch.arange(C, device=device).repeat(B)

            eps_pred = self.model(x_t_rep, t_rep, y_labels).view(B, C, -1)
            
            # Accumulate MSE across trials
            target_noise = noise.unsqueeze(1).expand(-1, C, -1)
            mse_accum += F.mse_loss(eps_pred, target_noise, reduction='none').mean(dim=2)

        # Average MSE across trials and convert to probabilities
        avg_mse = mse_accum / self.num_trials
        return F.softmax(-avg_mse, dim=1)  # Lower MSE = higher probability

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    model.train()
    device = next(model.parameters()).device
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            batch_size = x.size(0)

            # Sample timesteps
            # 1. Sample x_0 ~ q(x_0) (data loading)
            # 2. Sample t ~ Uniform({1,...,T})
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()

            # 3. Sample ϵ ~ N(0, I)
            epsilon = torch.randn_like(x)

            # Compute noisy x
            alpha_bars_t = alpha_bars[t]
            # x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ϵ
            x_t = torch.sqrt(alpha_bars_t)[:, None] * x + torch.sqrt(1 - alpha_bars_t)[:, None] * epsilon

            # Predict and compute loss
            epsilon_theta = model(x_t, t)
            # 4. Take gradient descent step on ∇_θ||ϵ - ϵ_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ϵ, t)||^2
            loss = F.mse_loss(epsilon_theta, epsilon)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        # Save model
        torch.save(model.state_dict(), f"{run_name}/model.pth")

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   
    model.eval()
    device = next(model.parameters()).device

    # Precompute scheduler parameters
    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Initialize noise
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    intermediates = []

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_tensor)

        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute mean and variance
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta) / sqrt_alpha_t
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(beta_t) * z

        x_t = x_prev

        if return_intermediate:
            intermediates.append(x_prev.cpu())

    return intermediates if return_intermediate else x_t

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name, p_uncond=0.1):
    model.train()
    device = next(model.parameters()).device
    alpha_bars = noise_scheduler.alpha_bars.to(device)
    null_class = model.n_classes  # Null class index

    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)

            # Randomly replace some labels with null class
            mask = torch.rand(batch_size, device=device) < p_uncond
            y[mask] = null_class # ỹ ← ∅ with probability p_uncond

            # Sample timesteps
            # 1. Sample x_0 ~ q(x_0) (data loading)
            # 2. Sample t ~ Uniform({1,...,T})
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
            
            # 3. Sample ϵ ~ N(0, I)
            epsilon = torch.randn_like(x)

            # Compute noisy x
            alpha_bars_t = alpha_bars[t]
            # x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ϵ
            x_t = torch.sqrt(alpha_bars_t)[:, None] * x + torch.sqrt(1 - alpha_bars_t)[:, None] * epsilon

            # Predict and compute loss
            epsilon_theta = model(x_t, t, y) # ϵ_θ(x_t, t, ỹ)
            # 4. Take gradient descent step on ∇_θ||ϵ - ϵ_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ϵ, t)||^2
            loss = F.mse_loss(epsilon_theta, epsilon)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        # Save model
        torch.save(model.state_dict(), f"{run_name}/model.pth")

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label, return_intermediate=False):
    """
    Pure conditional generation without guidance
    """
    model.eval()
    device = next(model.parameters()).device

    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Initialize with pure noise (x_T ∼ N(0,I))
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    
    intermediates = []

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise ε_θ(x_t, t, y) (Conditional model)
        epsilon_theta = model(x_t, t_tensor, torch.full((n_samples,), class_label, device=device, dtype=torch.long))

        # Reverse process step
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute μ_θ(x_t, t, y)
        # μ_θ = (x_t - β_t/√(1-ᾱ_t) * ε_θ) / √α_t
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta) / sqrt_alpha_t
        
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(beta_t) * z

        if return_intermediate:
            intermediates.append(x_prev.cpu().clone())  # Clone and move to CPU to save GPU memory

        x_t = x_prev

    if return_intermediate:
        return intermediates
    else:
        return x_t

@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    model.eval()
    device = next(model.parameters()).device

    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Initialize with pure noise (x_T ∼ N(0,I))
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    null_class = model.n_classes  # Null class index

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        # Duplicate inputs for conditional and unconditional
        x_t_repeated = torch.cat([x_t, x_t], dim=0)
        t_repeated = torch.cat([t_tensor, t_tensor], dim=0)
        class_labels = torch.cat([
            torch.full((n_samples,), class_label, device=device, dtype=torch.long),
            torch.full((n_samples,), null_class, device=device, dtype=torch.long),
        ], dim=0)

        # Predict noise for both conditions
        epsilon = model(x_t_repeated, t_repeated, class_labels)
        epsilon_cond, epsilon_uncond = torch.chunk(epsilon, 2, dim=0)

        # Combine using guidance scale
        # ϵ̃ = ϵ_uncond + guidance_scale*(ϵ_cond - ϵ_uncond)
        epsilon_theta = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        # Reverse process step
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])

        # Compute x_{t-1} mean
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta) / sqrt_alpha_t
        
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(beta_t) * z

        x_t = x_prev

    return x_t

def classifier_reward(x, target_class, classifier, temperature=1.0):
    """
    Reward function that scores samples based on classifier confidence
    
    Args:
        x: torch.Tensor [batch_size, n_dim] - Input samples
        target_class: int - Target class to maximize probability for
        classifier: ClassifierDDPM - Classifier model
        temperature: float - Controls sharpness of the reward signal
        
    Returns:
        torch.Tensor [batch_size] - Reward scores
    """
    # Get class probabilities from classifier
    probs = classifier.predict_proba(x)
    
    # Extract probability of target class
    target_probs = probs[:, target_class]
    
    # Apply temperature scaling and log transform for better gradient properties
    rewards = torch.log(target_probs + 1e-10) * temperature
    
    return rewards

@torch.no_grad()
def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model using reward-guided diffusion

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float, scale factor for reward guidance
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    model.eval()
    device = next(model.parameters()).device

    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)

    # Initialize with pure noise (x_T ∼ N(0,I))
    x_t = torch.randn(n_samples, model.n_dim, device=device)
    null_class = model.n_classes

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Get unconditional noise prediction
        epsilon_pred = model(x_t, t_tensor, torch.full((n_samples,), null_class, device=device, dtype=torch.long))
        
        # Compute the reverse process parameters
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])
        
        # Compute base posterior mean
        mu_t = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_pred) / sqrt_alpha_t
        
        if t > 0 and reward_scale > 0:
            # Estimate gradient of reward with respect to x_t using score matching
            x_t_grad = x_t.clone().detach().requires_grad_(True)
            
            # Forward pass through model to get epsilon prediction
            with torch.enable_grad():
                eps_t = model(x_t_grad, t_tensor, torch.full((n_samples,), null_class, device=device, dtype=torch.long))
                
                # Compute predicted x_{t-1} mean (posterior mean)
                mu_theta = (x_t_grad - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_t) / sqrt_alpha_t
                
                # Evaluate reward on predicted x_{t-1}
                rewards = reward_fn(mu_theta)
                
                # Backpropagate to get gradient
                reward_sum = rewards.sum()
                reward_sum.backward()
            
            # Extract gradient
            grad_x_t = x_t_grad.grad
            
            # Apply reward gradient to guide the sampling
            variance_t = beta_t
            grad_scale = reward_scale * variance_t
            mu_t = mu_t + grad_scale * grad_x_t
            
            z = torch.randn_like(x_t)
            x_t = mu_t + torch.sqrt(beta_t) * z
        else:
            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = mu_t + torch.sqrt(beta_t) * z

    return x_t


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.dataset}_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth', weights_only=True))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

    else:
        raise ValueError(f"Invalid mode {args.mode}")
