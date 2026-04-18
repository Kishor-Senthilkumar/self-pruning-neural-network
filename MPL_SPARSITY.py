
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class GateLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_features))
        self.g = nn.Parameter(torch.randn(out_features, in_features) - 2)

    def forward(self, x):
        gate  = torch.sigmoid(self.g * 4)
        w_eff = self.w * gate
        return F.linear(x, w_eff, self.b)

    def penalty(self):
        return torch.sum(torch.sigmoid(self.g * 4))

    def gate_values(self):
        with torch.no_grad():
            return torch.sigmoid(self.g * 4)


class SparseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1  = GateLayer(3072, 800)
        self.l2  = GateLayer(800, 400)
        self.l3  = GateLayer(400, 200)
        self.l4  = GateLayer(200, 10)
        self.bn1 = nn.BatchNorm1d(800)
        self.bn2 = nn.BatchNorm1d(400)
        self.bn3 = nn.BatchNorm1d(200)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = F.relu(self.bn3(self.l3(x)))
        return self.l4(x)

    def sparsity_term(self):
      total = 0
      for m in self.modules():
         if isinstance(m, GateLayer):
             total += torch.sum(torch.sigmoid(m.g * 4))
      return total

    def sparsity_ratio(self, threshold=1e-2):
      total = zero = 0
      for m in self.modules():
          if isinstance(m, GateLayer):
             g = torch.sigmoid(m.g * 10)
             total += g.numel()
             zero += (g < threshold).sum().item()
      return zero / total
    def all_gate_values(self):
        parts = []
        for m in self.modules():
            if isinstance(m, GateLayer):
                parts.append(m.gate_values().cpu().numpy().ravel())
        return np.concatenate(parts)


def load_data(batch_size=256):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR10(
        "./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_tf)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=False),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=False))


def train_epoch(net, loader, optimizer, lam):
    net.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = net(x)
        loss = F.cross_entropy(out, y) + lam * net.sparsity_term()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (net(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def plot_distributions(all_results):
    n    = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    colors = ["#378ADD", "#1D9E75", "#D85A30"]
    for ax, res, color in zip(axes, all_results, colors):
        ax.hist(res["gates"], bins=80, range=(0, 1),
                color=color, edgecolor="none", alpha=0.85)
        ax.axvline(0.01, color="#E24B4A", linestyle="--",
                   linewidth=1.3, label="prune threshold")
        ax.set_title(
            f"lambda = {res['lam']:.0e}\n"
            f"Acc: {res['acc']*100:.1f}%  Sparsity: {res['sparsity']*100:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved: gate_distributions.png")
    plt.close()


def run():
    train_loader, test_loader = load_data()
    lambdas     = [1e-5, 5e-5, 1e-4]
    epochs      = 40
    all_results = []

    for lam in lambdas:
        print(f"\n{'='*50}")
        print(f"  Lambda = {lam:.0e}")
        print(f"{'='*50}")

        model = SparseMLP().to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for ep in range(1, epochs + 1):
            loss, tr_acc = train_epoch(model, train_loader, opt, lam)
            val_acc      = evaluate(model, test_loader)
            sparsity     = model.sparsity_ratio()
            sched.step()
            print(f"  Ep {ep:02d}/{epochs} | Loss {loss:.3f} | "
                  f"Train {tr_acc*100:.1f}% | Test {val_acc*100:.1f}% | "
                  f"Sparsity {sparsity*100:.1f}%")

        final_acc      = evaluate(model, test_loader)
        final_sparsity = model.sparsity_ratio()
        all_results.append({
            "lam"      : lam,
            "acc"      : final_acc,
            "sparsity" : final_sparsity,
            "gates"    : model.all_gate_values(),
        })

    print(f"\n{'='*50}")
    print(f"  {'Lambda':<12} {'Accuracy':>12} {'Sparsity':>12}")
    print(f"  {'-'*38}")
    for r in all_results:
        print(f"  {r['lam']:<12.0e} {r['acc']*100:>11.2f}% "
              f"{r['sparsity']*100:>11.2f}%")
    print(f"{'='*50}")

    plot_distributions(all_results)


if __name__ == "__main__":
    run()