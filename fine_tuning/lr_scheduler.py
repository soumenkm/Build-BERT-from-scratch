import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, eta_min=1e-8)

lr_list = []
for ep in range(10):
    for i in range(100):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep}, Step: {i}, lr: {lr}")
        lr_list.append(lr)
        scheduler.step(ep+i/100)

plt.figure(figsize=(10, 6))
plt.plot(lr_list)
plt.title('Learning Rate Schedule with Cosine Annealing Warm Restarts')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.savefig('lr_schedule_plot.png')
plt.show()
