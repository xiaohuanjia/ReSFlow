import argparse
import os
import time
from configuration import parse_run_args

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import get_flow_model
from datasets import get_dataset
from utils import seed_all, load_config, get_optimizer, get_scheduler, count_parameters, recursive_to_device


def collect_cond_args_from_loader(loader, n_sample, device):
    """Take up to n_sample condition tensors from a DataLoader (same layout as *cond_args in train)."""
    out = None
    n = 0
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError('conditioned dataset batch must be (x, *cond_args)')
        _, *cond_args = batch
        cond_args = recursive_to_device(cond_args, device)
        take = min(n_sample - n, cond_args[0].size(0))
        if take <= 0:
            break
        if out is None:
            out = [c[:take] for c in cond_args]
        else:
            for i in range(len(cond_args)):
                out[i] = torch.cat([out[i], cond_args[i][:take]], dim=0)
        n += take
        if n >= n_sample:
            break
    if out is None:
        raise ValueError('Failed to collect conditions from loader: data is empty')
    return tuple(out)
from visualize import get_vis
from reflow import run_reflow_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'inf', 'reflow'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--method', choices=['euler', 'ode'], default='ode')
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--max_batch', type=int, default=None)
    args = parse_run_args(parser)
    if args.mode == 'inf' and args.resume is None:
        parser.error('Inference requires run.resume in YAML or --resume')
    if args.resume is not None and not os.path.isfile(args.resume):
        parser.error(f'Checkpoint missing: {args.resume}. Run the preceding training stage or update run.resume.')

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    train_set, valid_set, test_set = get_dataset(config.datasets)
    visualizer = get_vis(config.visualizer, writer, args.device, data_root=config.datasets.root)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)

    # Model
    print('Building model...')
    model = get_flow_model(config.model, config.encoder).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            # pytorch-lightning ckpt: Text8Module keys look like 'model.xxx'
            sd = ckpt['state_dict']
            if any(k.startswith('model.') for k in sd.keys()):
                state_dict = {k[len('model.'):]: v for k, v in sd.items()
                              if k.startswith('model.')}
            else:
                state_dict = sd
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'  [resume] missing keys: {len(missing)} (e.g. {missing[:3]})')
        if unexpected:
            print(f'  [resume] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})')
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for x, *cond_args in train_loader:
                # Training
                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1

                # Validation
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate(valid_loader)
                    sample('euler', 'valid', config.get('sample_max_batch', None))
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate(dataloader, split='valid'):
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.get('valid_max_batch', None)
            if total is None:
                total = len(dataloader)
            for i, (x, *cond_args) in tqdm(enumerate(dataloader), total=total):
                if i >= total:
                    break
                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar(f'{split}/loss', val_loss, global_step)
        print(f'Step {global_step} {split} loss {val_loss:.6f}')
        return val_loss


    def sample(method='euler', split='valid', max_batch=None):
        with torch.no_grad():
            model.eval()
            if not config.conditioned:
                traj = visualizer(model, method, global_step)
            else:
                dataloader = valid_loader if split == 'valid' else test_loader
                traj = visualizer(model, dataloader, method, global_step, max_batch=max_batch)
        return traj


    try:
        if args.mode == 'train':
            train()
            print('Training finished!')
            sample('ode', 'valid', None)
            print('Sampling finished!')

        elif args.mode == 'reflow':
            if args.resume is None:
                print('No base checkpoint provided, training base model first...')
                train()
                print('Base training finished!')
            print('\nStarting Reflow pipeline...')
            result = run_reflow_pipeline(
                model, config, args.device,
                writer=writer, logdir=logdir,
                dataloader=train_loader if config.conditioned else None,
                valid_loader=valid_loader,
            )
            reflow_model = result['reflow_model']
            print('\nReflow sampling (1-step Euler):')
            reflow_model.eval()
            n_vis = 100
            cond_vis = ()
            if config.conditioned:
                cond_vis = collect_cond_args_from_loader(valid_loader, n_vis, args.device)
                n_vis = cond_vis[0].size(0)
            with torch.no_grad():
                traj = reflow_model.sample('euler', n_vis, 1, args.device, *cond_vis)
            visualizer.writer = writer
            if hasattr(visualizer, 'n_step'):
                visualizer.n_step = 1
            from torchvision.utils import make_grid
            if config.visualizer.type == 'bmnist':
                img = (traj[..., 0] > traj[..., 1]).float().view(-1, 28, 28)
                img = make_grid(img.unsqueeze(1), nrow=10, padding=2, pad_value=0)
                writer.add_image('reflow_sample_1step', img, 0)
            elif config.visualizer.type == 'text8':
                TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")
                ids = traj.argmax(-1).tolist()
                for i, row in enumerate(ids[:min(8, len(ids))]):
                    writer.add_text(f'reflow_sample_1step/{i}',
                                    ''.join(TEXT8_CHARS[j] for j in row), 0)

            print('Reflow pipeline finished!')

        elif args.mode == 'inf':
            split = args.split
            method_inf = args.method
            n_steps_inf = args.n_steps if args.n_steps is not None else config.visualizer.get('n_step', 300)
            visualizer.n_step = n_steps_inf
            print(f'Inference: method={method_inf}, split={split}, n_steps={n_steps_inf}')
            sample(method_inf, split, args.max_batch)
            print(f'Sampling finished! (method={method_inf}, split={split}, n_steps={n_steps_inf})')

        time.sleep(3)
    except KeyboardInterrupt:
        print('Terminating...')
