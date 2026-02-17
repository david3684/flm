import json
import os
import itertools
import functools
import argparse
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
torch.load = functools.partial(torch.load, weights_only=False)
from torch.distributed import init_process_group, destroy_process_group
import wandb
import algo
import dataloader
import utils

import numpy as np
from datetime import datetime

import uuid

# Allow torch.load(weights_only=True) to safely unpickle Hydra configs stored in checkpoints
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Metadata])

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if 'hf' in config.algo.backbone:
        return diffusion_model(
            config, tokenizer=tokenizer).to('cuda')

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
            ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                   'entropy': entropy,
                   'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)


def _generate_samples_with_tc(diffusion_model, config, logger,
                              tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for i in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            assert config.loader.eval_batch_size % config.sampling.duplicate == 0
            different_in_batch = config.loader.eval_batch_size // config.sampling.duplicate
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps, duplicate=config.sampling.duplicate)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            model.metrics.record_tc([i*different_in_batch + j for _ in range(
                config.sampling.duplicate) for j in range(different_in_batch)], samples)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        avg_tc, avg_joints, avg_marginals = model.metrics.tc.compute()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
        print('Total average correlation:', avg_tc)
        print('Average joint entropy:', avg_joints)
        print('Average marginal entropy:', avg_marginals)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                   'entropy': entropy,
                   'avg_tc': avg_tc,
                   'avg_joints': avg_joints,
                   'avg_marginals': avg_marginals,
                   'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)


@torch.inference_mode()
def generate_reflow_dataset(diffusion_model, config, logger, tokenizer):
    # TODO: implement with lightning_module.test with pseudo-data
    logger.info('Generating samples.')
    model = _load_from_checkpoint(diffusion_model=diffusion_model,
                                  config=config,
                                  tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    # if model.ema:
    #   model.ema.store(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.ema.copy_to(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.backbone.eval()
    #   model.noise.eval()

    test_ds = dataloader.get_pseudo_dataloader(config, tokenizer, model)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=None,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=None)
    trainer.test(model, test_ds)
    return


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Perplexity Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


@torch.inference_mode()
def generate_reflow_dataset_with_perturbed_rect(diffusion_model, config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    train_ds, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True)

    # i is given by random sequence of train_ds's N
    shuffled_indices = np.random.permutation(len(train_ds.dataset))

    eval_batch_size = config.loader.eval_batch_size
    generate_samples = config.sampling.num_reflow_samples

    x0s = []
    xTs = []
    ts = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for j in range(generate_samples // eval_batch_size):
        if config.sampling.semi_ar:
            raise NotImplementedError(
                "Semi-AR sampling is not implemented. Please use standard sampling.")
        else:
            assert eval_batch_size == 1
            x0 = train_ds.dataset[shuffled_indices[j *
                                                   eval_batch_size:(j+1)*eval_batch_size]]['input_ids']
            x0 = torch.from_numpy(x0).to(model.device)
            x1 = torch.randint(0, 50258, x0.shape,
                               device=model.device, dtype=x0.dtype)
            rand_t = torch.randint(
                0, x0.shape[1], (1, ), device=model.device).float().item() / x0.shape[1]
            num_step = max(int(config.sampling.steps * (1 - rand_t)), 1)
            # random interpolate between x1 and x0
            # =y_given_t where y0=noise, y1=data
            xt = torch.where(rand_t > torch.rand(
                x1.shape, device=model.device), x0, x1)

            samples = model.restore_model_and_sample(
                num_steps=num_step, xT=xt.clone(), given_t=rand_t)
            x0s.append(samples.clone())
            xTs.append(xt.clone())
            ts.append(rand_t)
        if j % 500 == 0:
            print(f"Generated {(j+1) * eval_batch_size} samples")
    x0s = torch.cat(x0s, dim=0)
    xTs = torch.cat(xTs, dim=0)
    ts = torch.tensor(ts, device=model.device)

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    x0s = x0s.cpu().numpy()
    xTs = xTs.cpu().numpy()
    ts = ts.cpu().numpy()

    save_path = config.data.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    xT_path = os.path.join(save_path, 'xT.npy')
    x0_path = os.path.join(save_path, 'x0.npy')
    ts_path = os.path.join(save_path, 'ts.npy')

    np.save(x0_path, x0s)
    np.save(xT_path, xTs)
    np.save(ts_path, ts)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wid = config.wandb.get('id')
        if not wid or len(str(wid)) > 16:
            wid = str(uuid.uuid4().hex[:8])
        config.wandb.id = wid
        if config.wandb.get('name'):
            config.wandb.name = f"{config.wandb.name}_{wid}"
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer,
            config=config,
            weights_only=False)
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    # Force weights_only=False to allow full checkpoint restore (PyTorch 2.6 defaults torch.load to weights_only=True)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == 'ar':
        diffusion_model = algo.AR
    elif config.algo.name == 'mdlm':
        diffusion_model = algo.MDLM
    elif config.algo.name == 'duo_base':
        diffusion_model = algo.DUO_BASE
    elif config.algo.name == 'duo':
        diffusion_model = algo.DUO
    elif config.algo.name == 'flm':
        diffusion_model = algo.FLM
    elif config.algo.name == 'flm_distill':
        diffusion_model = algo.FLM_distill
    elif config.algo.name == 'flm_distill_double':
        diffusion_model = algo.FLM_distill_double
    elif config.algo.name == 'd3pm':
        diffusion_model = algo.D3PMAbsorb
    elif config.algo.name == 'sedd':
        diffusion_model = algo.SEDDAbsorb
    elif config.algo.name == 'dmf':
        diffusion_model = algo.DMF
    elif config.algo.name == 'distillation':
        diffusion_model = algo.Distillation
    elif config.algo.name == 'ot-finetune':
        diffusion_model = algo.OptimalTransportFinetune
    elif config.algo.name == 'rectification':
        diffusion_model = algo.Rectification
    elif config.algo.name == 'DMF':
        diffusion_model = algo.DMF
    elif config.algo.name == 'continuousflow':
        diffusion_model = algo.ContinuousFlow
    else:
        raise ValueError(
            f'Invalid algorithm name: {config.algo.name}')
    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_recon':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_with_tc':
        _generate_samples_with_tc(**kwargs)
    elif config.mode == 'ppl_eval':
        _eval_ppl(**kwargs)
    elif config.mode == 'generate_reflow_data':
        generate_reflow_dataset(diffusion_model, config, logger, tokenizer)
    elif config.mode == 'generate_reflow_data_with_perturbed_rect':
        generate_reflow_dataset_with_perturbed_rect(**kwargs)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
