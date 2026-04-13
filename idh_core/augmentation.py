import numpy as np
import torch
import torch.optim as optim

from .models import DEVICE, ConditionalGenerator, TimeSeriesDiscriminator, compute_gradient_penalty


def train_wgan_gp(train_data, latent_dim, seq_len, num_features, epochs=20, batch_size=64, n_critic=3):
    train_data = train_data.float().to(DEVICE)
    train_data = (train_data - train_data.mean()) / (train_data.std() + 1e-8)

    generator = ConditionalGenerator(latent_dim, 2, seq_len, num_features).to(DEVICE)
    discriminator = TimeSeriesDiscriminator(seq_len, num_features).to(DEVICE)
    optim_g = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))

    for _ in range(epochs):
        indices = torch.randperm(train_data.size(0), device=train_data.device)
        train_data = train_data[indices]
        for start in range(0, train_data.size(0), batch_size):
            for _ in range(n_critic):
                real_data = train_data[start : start + batch_size]
                current_bs = real_data.size(0)
                if current_bs == 0:
                    continue
                optim_d.zero_grad()
                noise = torch.randn(current_bs, latent_dim, device=DEVICE)
                fake_data = generator(noise, torch.zeros(current_bs, device=DEVICE).long())
                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data.detach())
                gp = compute_gradient_penalty(discriminator, real_data.data, fake_data.data, DEVICE)
                loss_d = -torch.mean(d_real) + torch.mean(d_fake) + 10 * gp
                loss_d.backward()
                optim_d.step()

            optim_g.zero_grad()
            noise = torch.randn(current_bs, latent_dim, device=DEVICE)
            generated = generator(noise, torch.zeros(current_bs, device=DEVICE).long())
            loss_g = -torch.mean(discriminator(generated))
            loss_g.backward()
            optim_g.step()
    return generator


def generate_samples(generator, num_samples, latent_dim, seq_len, num_features, labels=None):
    if labels is None:
        labels = torch.zeros(num_samples, dtype=torch.long, device=DEVICE)
    z = torch.randn(num_samples, latent_dim, device=DEVICE)
    with torch.no_grad():
        samples = generator(z, labels).detach().cpu()
    return samples


def augment_positive_class(train_time_x, train_labels, latent_dim=128):
    train_time_x = torch.as_tensor(train_time_x, dtype=torch.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    positive_indices = np.where(train_labels == 1)[0]
    if len(positive_indices) == 0:
        return train_time_x, torch.as_tensor(train_labels)

    positive_data = train_time_x[positive_indices]
    generator = train_wgan_gp(
        positive_data,
        latent_dim=latent_dim,
        seq_len=positive_data.shape[1],
        num_features=positive_data.shape[2],
        epochs=1,
    )
    synthetic = generate_samples(
        generator,
        num_samples=len(positive_indices),
        latent_dim=latent_dim,
        seq_len=positive_data.shape[1],
        num_features=positive_data.shape[2],
        labels=torch.ones(len(positive_indices), dtype=torch.long, device=DEVICE),
    )
    augmented_x = torch.cat([train_time_x, synthetic], dim=0)
    augmented_y = torch.cat([torch.as_tensor(train_labels), torch.ones(len(positive_indices), dtype=torch.long)], dim=0)
    return augmented_x, augmented_y
