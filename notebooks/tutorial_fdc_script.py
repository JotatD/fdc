import copy

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genexp.models import DiffusionModel
from genexp.sampling import VPSDE, EulerMaruyamaSampler
from genexp.trainers.genexp import FDCTrainerFlow
from genexp.utils import seed_everything


class LightningDiffusion(LightningModule):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (x0,) = batch
        t = torch.rand(x0.shape[0]).to(x0.device)
        alpha, sig = self.model.sde.get_alpha_sigma(t[:, None])
        eps = torch.randn(x0.shape).to(x0.device)

        xt = torch.sqrt(alpha) * x0 + sig * eps

        eps_pred = self(xt, t[:, None])

        loss = torch.mean((eps - eps_pred) ** 2) / 2.0
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


x0 = torch.randn((50000, 2))
x1 = torch.randn((5000, 2)) * 0.3 + 3
dataset = torch.vstack((x0, x1))
network = nn.Sequential(
    nn.Linear(3, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)

sde = VPSDE(0.1, 12)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = DiffusionModel(network, sde).to(device)
pl_model = LightningDiffusion(model)


dl = DataLoader(TensorDataset(dataset), batch_size=128, shuffle=True)

# trainer = Trainer(max_epochs=10)
# trainer.fit(pl_model, dl)

# # Create output directories
# os.makedirs("models", exist_ok=True)
# os.makedirs("figs", exist_ok=True)

# # Save model
# torch.save(model.model.state_dict(), "models/gauss_model.pth")

# Load model
model.model.load_state_dict(torch.load("models/gauss_model.pth", map_location=device))

# Visualize pre-trained density
# sampler = EulerMaruyamaSampler(model.to(device), data_shape=(2,), device=device)
# samples = []
batch_size = 256
num_samples = 10000
# for i in tqdm(range(num_samples // batch_size + 1)):
#     trajs, ts = sampler.sample_trajectories(N=batch_size, T=1000, device=device)
#     samples.append(trajs[-1].full.detach().cpu())
# samples = torch.vstack(samples)[:num_samples]

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].hist(dataset[:, 0], bins=150)
# ax[1].hist(samples[:, 0].detach().cpu(), bins=150)
# ax[0].set_title("Data density")
# ax[1].set_title("Pre-trained model density")
# plt.tight_layout()
# fig.savefig("figs/pretrained_density.png")
# plt.close(fig)

# Fine-tuning with FDC
config = OmegaConf.load("../configs/example_fdc.yaml")
sampler = EulerMaruyamaSampler(model, data_shape=(2,), device=device)
model = model.to(device)
seed_everything(config.seed)
print("Next is the flow density control fine-tuning...")
fdc_trainer = FDCTrainerFlow(
    config, copy.deepcopy(model), copy.deepcopy(model), device=device, sampler=sampler
)
for k in tqdm(range(config.num_md_iterations)):
    for i in range(config.adjoint_matching.num_iterations):
        am_dataset = fdc_trainer.generate_dataset()
        fdc_trainer.finetune(am_dataset, steps=config.adjoint_matching.finetune_steps)
    fdc_trainer.update_base_model()

# Visualize fine-tuned model's density
sampler = EulerMaruyamaSampler(
    fdc_trainer.fine_model.to(device), data_shape=(2,), device=device
)
samples_fdc = []
for i in tqdm(range(num_samples // batch_size + 1)):
    trajs, ts = sampler.sample_trajectories(N=batch_size, T=1000, device=device)
    samples_fdc.append(trajs[-1].full.detach().cpu())
samples_fdc = torch.vstack(samples_fdc)[:num_samples]

# fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# ax[0].hist(dataset[:, 0], bins=150)
# ax[1].hist(samples[:, 0].detach().cpu(), bins=100)
# ax[2].hist(samples_fdc[:, 0].detach().cpu(), bins=100)
# ax[0].set_title("Data density")
# ax[1].set_title("Pre-trained model density")
# ax[2].set_title("Fine-tuned model density")
# plt.tight_layout()
# fig.savefig("figs/finetuned_density.png")
# plt.close(fig)
