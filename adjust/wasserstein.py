import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cpu import device_count
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import sys
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Function


def print_now(*args, **kwargs):
    print(*args, flush=True, **kwargs)


class MLP(nn.Module):
    def __init__(self, input_dim, n_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden):
        super(Critic, self).__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.hidden = MLP(hidden_dim, n_hidden)
        # The final layer must output a single scalar value.
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = self.hidden(x)
        x = self.final(x)
        # Return the raw score
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_hidden=4, n_classes=2):
        super(Classifier, self).__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        # We need a separate MLP instance for this classifier
        self.hidden = MLP(hidden_dim, n_hidden)
        self.final = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = self.hidden(x)
        x = self.final(x)
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_dim, encoder_layers, hidden_dim, classifier_layers, n_classes):
        super(SimpleNN, self).__init__()
        self.mlp = MLP(input_dim, encoder_layers)
        self.critic = Critic(input_dim, hidden_dim, classifier_layers)
        self.monitoring_classifier = Classifier(input_dim, hidden_dim=128, n_hidden=4, n_classes=n_classes)

    def forward(self, x):
        encoded_x = x + self.mlp(x)
        critic_score = self.critic(encoded_x)
        return encoded_x, critic_score


def gradient_penalty(critic, group0_features, group1_features, device="cpu"):
    """
    Calculates the gradient penalty loss for WGAN-GP.
    It penalizes the critic for having a gradient norm other than 1.
    """
    # Get random interpolation weight
    epsilon = torch.rand(group0_features.size(0), 1).to(device)
    epsilon = epsilon.expand(group0_features.size())

    # Create interpolated features
    interpolated_features = epsilon * group0_features + ((1 - epsilon) * group1_features)
    interpolated_features.requires_grad_(True)

    # Calculate critic scores on interpolated features
    critic_scores = critic(interpolated_features)

    # Take the gradient of the scores with respect to the features
    gradients = torch.autograd.grad(
        inputs=interpolated_features,
        outputs=critic_scores,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


def train(model, train_loader, main_optimizer, critic_optimizer, monitoring_optimizer, epochs, device, verbose=False,
          lambda_gp=10, critic_iterations=5, monitor_iterations=5):
    model.train()
    for epoch in range(epochs):
        all_critic_loss = []
        all_reconstruction_loss = []
        all_adversarial_loss = []
        all_monitor_loss = []
        num_monitor_correct = 0

        p_remove = 0.2

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            mask = torch.ones_like(x_batch)
            mask[torch.rand_like(x_batch) <= p_remove] = 0.0
            masked_x_batch = x_batch * mask

            # Get the encoder's output once for this batch
            encoded_x, _ = model(masked_x_batch)
            # Detach it now for use in both critic and monitor training
            detached_encoded_x = encoded_x.detach()

            # --- Step 1: Train the Critic for `critic_iterations` ---
            for critic_iter in range(critic_iterations):
                critic_optimizer.zero_grad()

                group0_features = detached_encoded_x[y_batch == 0]
                group1_features = detached_encoded_x[y_batch == 1]

                if group0_features.nelement() > 0 and group1_features.nelement() > 0:
                    min_size = min(group0_features.size(0), group1_features.size(0))
                    group0_features = group0_features[:min_size]
                    group1_features = group1_features[:min_size]

                    critic_scores_g0 = model.critic(group0_features)
                    critic_scores_g1 = model.critic(group1_features)

                    wasserstein_distance = critic_scores_g1.mean() - critic_scores_g0.mean()
                    gp = gradient_penalty(model.critic, group0_features, group1_features, device)
                    critic_loss = -wasserstein_distance + lambda_gp * gp

                    critic_loss.backward()
                    critic_optimizer.step()

                    if critic_iter == critic_iterations - 1:
                        all_critic_loss.append(-wasserstein_distance.item())

            # --- Step 2: Train the Monitoring Classifier for `monitor_iterations` ---
            for monitor_iter in range(monitor_iterations):
                monitoring_optimizer.zero_grad()
                monitor_pred = model.monitoring_classifier(detached_encoded_x)
                monitor_loss = F.cross_entropy(monitor_pred, y_batch)
                monitor_loss.backward()
                monitoring_optimizer.step()

            # After all monitor training, get the final loss and accuracy for logging
            final_monitor_pred = model.monitoring_classifier(detached_encoded_x)
            final_monitor_loss = F.cross_entropy(final_monitor_pred, y_batch)
            all_monitor_loss.append(final_monitor_loss.item())
            num_monitor_correct += final_monitor_pred.argmax(dim=1).eq(y_batch).sum().item()

            # --- Step 3: Train the Encoder (run once per batch) ---
            main_optimizer.zero_grad()

            # Re-evaluate with graph attached for the encoder
            encoded_x_for_encoder, critic_scores = model(masked_x_batch)
            scores_g0 = critic_scores[y_batch == 0]
            scores_g1 = critic_scores[y_batch == 1]
            reconstruction_loss = F.mse_loss(encoded_x_for_encoder, x_batch)

            adversarial_weight = 10.0

            if scores_g0.nelement() > 0 and scores_g1.nelement() > 0:
                encoder_adversarial_loss = scores_g1.mean() - scores_g0.mean()
                mlp_total_loss = reconstruction_loss + adversarial_weight * encoder_adversarial_loss
                all_adversarial_loss.append(encoder_adversarial_loss.item())
            else:
                mlp_total_loss = reconstruction_loss

            mlp_total_loss.backward()
            main_optimizer.step()
            all_reconstruction_loss.append(reconstruction_loss.item())

        if verbose:
            avg_c_loss = sum(all_critic_loss) / len(all_critic_loss) if all_critic_loss else 0
            avg_a_loss = sum(all_adversarial_loss) / len(all_adversarial_loss) if all_adversarial_loss else 0
            avg_r_loss = sum(all_reconstruction_loss) / len(all_reconstruction_loss)
            monitor_acc = num_monitor_correct / len(train_loader.dataset)
            print_now(
                f"Epoch {epoch + 1}/{epochs}: "
                f"critic_dist={avg_c_loss:.4f}, "
                f"encoder_adv_loss={avg_a_loss:.4f}, "
                f"recon_loss={avg_r_loss:.4f}, "
                f"monitor_acc={monitor_acc:.4f}"
            )


def main(input_file, batch_col, output_file="output.csv", verbose=False, epochs=100):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print_now(f"Using device: {device}")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print_now(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    meta_cols = [col for col in df.columns if col.startswith('meta_')]
    feature_cols = [col for col in df.columns if not col.startswith('meta_') and col != batch_col]
    if not feature_cols:
        print_now("Error: No feature columns found.")
        sys.exit(1)

    X = df[feature_cols].values
    y_raw = df[batch_col].values

    print(f"Baseline accuracy: {df[batch_col].value_counts(normalize=True).max()}")

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print(f"Mean: {means.mean():.4f}, std: {stds.mean():.4f}")

    #X = (X - means) / stds

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_features = X.shape[1]
    n_classes = len(le.classes_)

    simplenn = SimpleNN(input_dim=n_features, encoder_layers=2, hidden_dim=128, classifier_layers=4,
                        n_classes=n_classes).to(device)

    # Main adversarial training optimizers
    main_optimizer = optim.Adam(simplenn.mlp.parameters(), lr=0.0001, betas=(0.5, 0.9))
    critic_optimizer = optim.Adam(simplenn.critic.parameters(), lr=0.0001, betas=(0.5, 0.9))

    # Optimizer for the monitoring classifier ONLY
    monitoring_optimizer = optim.Adam(simplenn.monitoring_classifier.parameters(), lr=0.001)

    train_dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Pass the new optimizer to the train function
    train(simplenn, train_loader, main_optimizer, critic_optimizer, monitoring_optimizer,
          epochs=epochs, verbose=verbose, device=device, lambda_gp=10,
          critic_iterations=5, monitor_iterations=5)

    # Save the modified features
    simplenn.eval()
    with torch.no_grad():
        X_test = torch.from_numpy(X).float().to(device)
        new_x, _ = simplenn(X_test)

        new_df = pd.DataFrame(new_x.cpu().numpy(), columns=feature_cols)
        new_df[batch_col] = y_raw
        new_df[meta_cols] = df[meta_cols]
        # Reorder columns
        new_df = new_df[meta_cols + [batch_col] + feature_cols]
        new_df.to_csv(output_file, index=False)
        print_now(f"Output saved to {output_file}")



if __name__ == "__main__":
    parser = ArgumentParser("Adversarial De-biasing PyTorch Classifier")
    parser.add_argument("-i", "--input-file", help="Path to input CSV file.", required=True)
    parser.add_argument("-b", "--batch-col", help="Column name for the sensitive attribute (target).", required=True)
    parser.add_argument("-o", "--output-file", help="Path to output CSV file.", required=False, default="output.csv")
    parser.add_argument("-v", "--verbose", help="Enable verbose output.", action="store_true")
    parser.add_argument("-e", "--epochs", help="Number of epochs to train.", type=int, default=100)
    args = parser.parse_args()
    main(args.input_file, args.batch_col, args.output_file, args.verbose, args.epochs)