import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

# ------------------ Candidate Network ------------------
class EvoNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ Mutation Engine ------------------
def mutate(model, mutation_rate=0.1, sigma=0.1):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in new_model.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(p) * sigma
                p.add_(noise)
    return new_model

# ------------------ Fitness Function ------------------
def fitness(model, data):
    x, y = data
    pred = model(x)
    return -F.mse_loss(pred, y).item()

# ------------------ Evolutionary Trainer ------------------
class EvolutionEngine:
    def __init__(self, population_size, in_dim, hidden_dim, out_dim):
        self.population = [
            EvoNet(in_dim, hidden_dim, out_dim)
            for _ in range(population_size)
        ]
        self.data = (torch.randn(64, in_dim), torch.randn(64, out_dim))

    def step(self):
        scored = []

        for model in self.population:
            score = fitness(model, self.data)
            scored.append((score, model))

        scored.sort(key=lambda x: x[0], reverse=True)

        # keep top 20%
        survivors = [m for _, m in scored[:max(1, len(scored)//5)]]

        # reproduce + mutate
        new_population = []
        while len(new_population) < len(self.population):
            parent = random.choice(survivors)
            child = mutate(parent)
            new_population.append(child)

        self.population = new_population

        return scored[0][0]

# ------------------ Run Evolution ------------------
engine = EvolutionEngine(population_size=20, in_dim=16, hidden_dim=32, out_dim=8)

for gen in range(60):
    best = engine.step()
    print(f"[GEN {gen}] Best Fitness: {best:.4f}")
