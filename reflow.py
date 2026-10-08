"""
Reflow for Categorical Flow Matching.

Implements the reflow pipeline:
  Layer 1: Base SFM / LFM (standard Riemannian flow matching)
  Layer 2: Standard Reflow (retrain base FM objective on generated pairs)

All functions are generalized to work with any CategoricalFlow subclass
(LinearCategoricalFlow, SphereCategoricalFlow, SimplexCategoricalFlow).
"""

import os

import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torchdiffeq import odeint

from utils import seed_all, recursive_to_device


# --- Validation helper ------------------------------------------------------

@torch.no_grad()
def _validate_fm_loss(model, valid_loader, device, max_batch=None):
    """
    Reuse the base-training validation protocol: iterate over valid_loader,
    compute model.get_loss for each batch, and return the average. The unit
    matches base training's valid/loss (FM objective on the real valid set),
    but is not strictly comparable to reflow_round*/loss (which is computed
    on generated pairs).

    Returns None when valid_loader is empty.
    """
    if valid_loader is None:
        return None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batch = 0
    limit = max_batch if max_batch is not None else len(valid_loader)
    for i, batch in enumerate(valid_loader):
        if i >= limit:
            break
        if isinstance(batch, (list, tuple)):
            x, *cond_args = batch
        else:
            x, cond_args = batch, []
        x = x.to(device)
        cond_args = recursive_to_device(cond_args, device)
        loss = model.get_loss(x, *cond_args)
        total_loss += float(loss.item())
        n_batch += 1
    if was_training:
        model.train()
    if n_batch == 0:
        return None
    return total_loss / n_batch


# --- Pair Generation --------------------------------------------------------

@torch.no_grad()
def generate_pair_batch(model, n_sample, method, n_steps, device, cond_args=()):
    """
    Generate (z0, z1) pairs by sampling noise and running the model forward.
    z0, z1 are returned in simplex space (postprocessed).
    cond_args: tuple of condition tensors (e.g. signal for promoter), already on device.
    """
    z0 = model.sample_prior(n_sample, model.total_data_dim, model.n_class, device=device)
    p0 = model.preprocess(z0)

    if method == 'ode':
        ps = odeint(
            lambda t, p: model(t, p, *cond_args),
            p0,
            t=torch.linspace(0., 1., 2, device=device, dtype=torch.float),
            atol=1e-5, rtol=1e-5,
        )
        z1 = model.postprocess(model.proj_x(ps[-1]))
    elif method == 'euler':
        p = p0
        dt = 1.0 / n_steps
        ts = torch.linspace(0, 1, n_steps + 1, device=device)
        for t in ts[:-1]:
            pred_vf = model(torch.full((n_sample,), t, device=device), p, *cond_args)
            p = model.exp(p, pred_vf * dt, model.eps)
            p = model.proj_x(p)
        z1 = model.postprocess(p)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return z0, z1


def generate_and_save_pairs(model, save_dir, n_pairs, batch_size,
                            method, n_steps, device, seed=0, dataloader=None):
    """
    Generate n_pairs (z0, z1) pairs and save to save_dir/.
    If dataloader is provided (conditioned model), conditions are taken from it
    and saved as cond_0.npy, cond_1.npy, ... alongside z0/z1.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)

    z0_all, z1_all = [], []
    cond_all = []
    model.eval()

    if dataloader is not None:
        collected = 0
        for x, *cond_args in tqdm(dataloader, desc='Generating pairs'):
            if collected >= n_pairs:
                break
            bs = x.size(0)
            cond_args_dev = tuple(c.to(device) for c in cond_args)
            z0, z1 = generate_pair_batch(model, bs, method, n_steps, device,
                                         cond_args=cond_args_dev)
            z0_all.append(z0.cpu())
            z1_all.append(z1.cpu())
            for i, c in enumerate(cond_args):
                if len(cond_all) <= i:
                    cond_all.append([])
                cond_all[i].append(c.cpu())
            collected += bs

        for i, cond_list in enumerate(cond_all):
            cond_tensor = torch.cat(cond_list, 0)[:n_pairs]
            np.save(os.path.join(save_dir, f'cond_{i}.npy'), cond_tensor.numpy())
    else:
        n_batches = (n_pairs + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), desc='Generating pairs'):
            bs = min(batch_size, n_pairs - i * batch_size)
            z0, z1 = generate_pair_batch(model, bs, method, n_steps, device)
            z0_all.append(z0.cpu())
            z1_all.append(z1.cpu())

    z0_all = torch.cat(z0_all, 0)[:n_pairs]
    z1_all = torch.cat(z1_all, 0)[:n_pairs]
    np.save(os.path.join(save_dir, 'z0.npy'), z0_all.numpy())
    np.save(os.path.join(save_dir, 'z1.npy'), z1_all.numpy())
    print(f'Saved {n_pairs} pairs to {save_dir}/')
    return z0_all, z1_all


def load_pairs(save_dir):
    """Load previously saved (z0, z1) pairs and optional conditions."""
    z0 = torch.from_numpy(np.load(os.path.join(save_dir, 'z0.npy'))).float()
    z1 = torch.from_numpy(np.load(os.path.join(save_dir, 'z1.npy'))).float()
    cond_list = []
    i = 0
    while os.path.exists(os.path.join(save_dir, f'cond_{i}.npy')):
        cond_list.append(
            torch.from_numpy(np.load(os.path.join(save_dir, f'cond_{i}.npy'))).float()
        )
        i += 1
    return z0, z1, cond_list


# --- Layer 2: Standard Reflow ----------------------------------------------

def get_reflow_loss(model, s0_data, s1_data, cond_args=(), t_schedule='uniform', eps=1e-3):
    """
    Standard Reflow loss: same objective as base FM (vecfield), but trained on
    reflow-generated (z0, z1) pairs instead of random (noise, data) pairs.

    Both s0_data and s1_data must be in the **preprocessed** manifold space.

    Steps:
      1. (pt, vf) = vecfield(s0, s1, t)   interpolation + conditional velocity
      2. pred     = model(t, pt)
      3. loss     = ||pred - vf||^2_{Riemannian at pt}
    """
    flow_cls = type(model)
    B, device = s0_data.size(0), s0_data.device

    if t_schedule == 'uniform':
        t = torch.rand(B, device=device) * (1. - eps) + eps
    elif isinstance(t_schedule, int):
        t = (torch.randint(0, t_schedule, (B,), device=device).float()
             * (1. - eps) / t_schedule + eps)
    else:
        raise ValueError(f"Unknown t_schedule '{t_schedule}'.")

    pt, vf = flow_cls.vecfield(s0_data, s1_data, t[:, None], eps)
    pred_vf = model(t, pt, *cond_args)
    return flow_cls.norm2(pt, pred_vf - vf, eps).mean()


def train_reflow_round(model, z0_data, z1_data, reflow_cfg, device,
                       writer=None, logdir=None, round_idx=1,
                       max_grad_norm=100., vis_tag_prefix=None,
                       vis_freq=None, cond_data=None,
                       valid_loader=None, val_freq=None, val_max_batch=None):
    """
    Train one round of standard reflow.

    :param model: CategoricalFlow model (will be modified in-place)
    :param z0_data: noise samples in simplex space, Tensor (N, D, C)
    :param z1_data: data samples in simplex space, Tensor (N, D, C)
    :param reflow_cfg: EasyDict with reflow training hyperparameters
    :param device: torch device
    :param writer: optional TensorBoard writer
    :param logdir: checkpoint save directory
    :param round_idx: reflow round index (for logging)
    :param max_grad_norm: gradient clipping
    :param cond_data: list of condition tensors (N, ...) for conditioned models
    :return: trained model
    """
    flow_cls = type(model)
    s0_all = flow_cls.preprocess(z0_data)
    s1_all = flow_cls.preprocess(z1_data)

    lr = reflow_cfg.get('lr', 1e-4)
    batch_size = reflow_cfg.get('batch_size', 256)
    max_iter = reflow_cfg.get('max_iter', 50000)
    log_freq = reflow_cfg.get('log_freq', 500)
    save_freq = reflow_cfg.get('save_freq', 10000)
    t_schedule = reflow_cfg.get('t_schedule', 'uniform')
    if val_freq is None:
        val_freq = reflow_cfg.get('val_freq', log_freq)

    print(f'  [Round {round_idx}] Reflow mode: standard')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(s0_all)

    model.train()
    for step in range(max_iter):
        idx = torch.randperm(n)[:batch_size]
        s0_b = s0_all[idx].to(device)
        s1_b = s1_all[idx].to(device)
        cond_args = tuple(c[idx].to(device) for c in cond_data) if cond_data else ()

        optimizer.zero_grad()
        loss = get_reflow_loss(model, s0_b, s1_b, cond_args=cond_args, t_schedule=t_schedule)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if writer:
            writer.add_scalar(f'reflow_round{round_idx}/loss', loss.item(), step)
            writer.add_scalar(f'reflow_round{round_idx}/grad', grad_norm.item(), step)

        if step % log_freq == 0:
            print(f'  [Round {round_idx}] Step {step}/{max_iter}  loss={loss.item():.6f}')

        # FM val_loss on the real valid set (same scale as base training valid/loss).
        if (valid_loader is not None and val_freq is not None
                and step % val_freq == 0):
            val_loss = _validate_fm_loss(model, valid_loader, device, max_batch=val_max_batch)
            if val_loss is not None:
                if writer is not None:
                    writer.add_scalar(f'reflow_round{round_idx}/val_loss', val_loss, step)
                print(f'  [Round {round_idx}] Step {step}/{max_iter}  val_loss={val_loss:.6f}')

        # Optional 1-step sampling visualisation (e.g., BMNIST / text8).
        if (writer is not None and vis_freq is not None
                and step % vis_freq == 0):
            with torch.no_grad():
                model.eval()
                n_class = model.n_class
                tag = vis_tag_prefix or f'reflow_round{round_idx}_sample_1step'
                if n_class == 2:
                    # BMNIST: render a 28x28 binary grid.
                    traj = model.sample('euler', 100, 1, device)
                    img = (traj[..., 0] > traj[..., 1]).float().view(-1, 28, 28)
                    img = make_grid(img.unsqueeze(1), nrow=10, padding=2, pad_value=0)
                    writer.add_image(tag, img, step)
                elif n_class == 27:
                    # text8: write 3 generated sequences to TensorBoard.
                    TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")
                    traj = model.sample('euler', 3, 1, device)
                    ids = traj.argmax(-1).tolist()
                    for i, row in enumerate(ids):
                        writer.add_text(f'{tag}/{i}',
                                        ''.join(TEXT8_CHARS[j] for j in row), step)
                # Other tasks: skip to avoid unnecessary sampling overhead.
                model.train()

        if logdir and step > 0 and step % save_freq == 0:
            torch.save({
                'model': model.state_dict(),
                'step': step,
                'round': round_idx,
            }, os.path.join(logdir, f'reflow_r{round_idx}_{step}.pt'))

    if logdir:
        torch.save({
            'model': model.state_dict(),
            'round': round_idx,
        }, os.path.join(logdir, f'reflow_r{round_idx}_final.pt'))

    print(f'  [Round {round_idx}] Training done!')
    return model


# --- Full Pipeline ----------------------------------------------------------

def run_reflow_pipeline(base_model, config, device, writer=None, logdir=None,
                        dataloader=None, valid_loader=None):
    """
    Run the full Reflow pipeline:
      1. Generate pairs from the base model
      2. Train k rounds of standard reflow

    :param base_model: trained CategoricalFlow model (Layer 1)
    :param config: full config with 'reflow' section
    :param device: torch device
    :param writer: TensorBoard writer
    :param logdir: save directory
    :param dataloader: DataLoader for conditioned models (provides condition args)
    :param valid_loader: optional DataLoader used to compute periodic val_loss
        (base FM loss on the real valid set; same scale as base training valid/loss)
    :return: dict with 'reflow_model' and the list of per-round 'reflow_models'
    """
    from models import get_flow_model

    rcfg = config.reflow
    k = rcfg.get('k', 2)
    n_pairs = rcfg.get('n_pairs', 50000)
    gen_batch_size = rcfg.get('gen_batch_size', 256)
    gen_method = rcfg.get('gen_method', 'ode')
    gen_n_steps = rcfg.get('gen_n_steps', 100)
    seed = rcfg.get('seed', 0)
    pair_dir = rcfg.get('pair_dir', os.path.join(logdir, 'reflow_pairs'))
    vis_freq = rcfg.get('vis_freq', None)
    val_max_batch = config.get('valid_max_batch', None)

    reflow_models = [base_model]

    for r in range(1, k + 1):
        print(f'\n{"=" * 60}')
        print(f'  Standard Reflow  round {r} / {k}')
        print(f'{"=" * 60}')

        prev_model = reflow_models[r - 1]
        round_dir = os.path.join(pair_dir, f'round_{r}')

        if os.path.exists(os.path.join(round_dir, 'z0.npy')):
            print(f'  [Round {r}] Loading existing pairs from {round_dir}')
            z0_data, z1_data, cond_data = load_pairs(round_dir)
        else:
            print(f'  [Round {r}] Generating {n_pairs} pairs ({gen_method.upper()}) ...')
            z0_data, z1_data = generate_and_save_pairs(
                prev_model, round_dir,
                n_pairs, gen_batch_size,
                gen_method, gen_n_steps,
                device, seed,
                dataloader=dataloader,
            )
            _, _, cond_data = load_pairs(round_dir)
        print(f'  [Round {r}] z0 {z0_data.shape}  z1 {z1_data.shape}')

        print(f'  [Round {r}] Standard Reflow training ...')
        new_model = get_flow_model(config.model, config.encoder).to(device)
        new_model.load_state_dict(prev_model.state_dict())
        train_reflow_round(
            new_model, z0_data, z1_data, rcfg, device,
            writer=writer, logdir=logdir, round_idx=r,
            max_grad_norm=config.train.get('max_grad_norm', 100.),
            vis_tag_prefix=f'reflow_round{r}_sample_1step',
            vis_freq=vis_freq,
            cond_data=cond_data if cond_data else None,
            valid_loader=valid_loader,
            val_max_batch=val_max_batch,
        )
        reflow_models.append(new_model)

    reflow_model = reflow_models[-1]
    return {'reflow_model': reflow_model, 'reflow_models': reflow_models}
