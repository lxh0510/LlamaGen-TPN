# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import numpy as np
import sys
import setproctitle
setproctitle.setproctitle("liuxh")
sys.path.append(os.path.abspath("/mnt/disk0/liuxh/LlamaGen"))
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models
from autoregressive.models.tpn import LabelPropagation
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.nn as nn

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    #optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])

        #start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        #train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        train_steps = 0
        start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    #model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    for p in model.parameters():
        p.requires_grad_(False)

    # 初始化标签传播

    tpn_model = LabelPropagation(args, device).to(device)
    tpn_model.train()

    #labels = torch.randint(0, 16384, (args.global_batch_size, 256))
    #one_hot = F.one_hot(labels, num_classes=16384).float()

    #logits = nn.Parameter((torch.ones((args.global_batch_size, 256, 16384), dtype=torch.float32) / 16384).to(device), requires_grad=True)

    optimizer = torch.optim.Adam(tpn_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.weight_decay)


    # 获取每个token的embedding 列表
    embedding_layer = model.tok_embeddings
    # EI   (16384,768)
    all_token_embeddings = embedding_layer.weight.detach()

    # 获取先验概率
    freq_probs = np.load(args.freq_probs_path)
    freq_probs = torch.from_numpy(freq_probs).to(device)
    
    # prior_probs = freq_probs / torch.sum(freq_probs, dim=1, keepdim=True)
    # max_idx = torch.argmax(freq_probs, dim=1, keepdim=True)
    k = 3
    topk_vals, topk_idx = torch.topk(freq_probs, k, dim=1)
    prior_probs = torch.zeros_like(freq_probs, dtype=torch.float32).scatter_(1, topk_idx, 1.0)

    # WI   (256,16384)
    prior_probs = prior_probs.unsqueeze(0).expand(args.global_batch_size,-1,-1).detach()

    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    #labels = torch.randint(0, 16384, (args.global_batch_size, 256))
    #one_hot = F.one_hot(labels, num_classes=16384).float()

    #logits = nn.Parameter(one_hot.to(device), requires_grad=True)

    #logits = nn.Parameter((torch.ones((args.global_batch_size, 256, 16384), dtype=torch.float32) / 16384).to(device), requires_grad=True)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            scheduler.step(train_steps)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, _, hidden_states = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)

            # 获得已生成token的embedding

            i = torch.randint(0, (256-args.topk), (1,)).item()
            generated_tokens = z_indices[: , :(i+1)]
            generated_embeddings = embedding_layer(generated_tokens)

            # 未生成token的embedding使用先验概率组合

            prior_embeddings = torch.matmul(prior_probs, all_token_embeddings)
            ungenerated_embeddings  = prior_embeddings[:,(i+1):,:]

            # 获取lebals

            labels = z_indices.clone()
            labels[:, (i+1):] = args.vocab_size

            # 标签传播获得logits

            logits = tpn_model(generated_embeddings, ungenerated_embeddings, hidden_states[:,:(i+1),:], labels)

            # 找出分布最尖锐的位置

            probs = torch.softmax(logits, dim=-1)

            # 使用置信度方法
    
            #print("z_labels.requires_grad:", z_labels.requires_grad)
            #print("pred_logits.requires_grad:", pred_logits.requires_grad)

            # 计算 loss   交叉熵函数包括softmax之前不能再softmax

            #loss = F.cross_entropy(pred_logits.reshape(args.global_batch_size * args.topk, -1), z_labels.reshape(args.global_batch_size * args.topk))
            max_probs, _ = probs.max(dim=-1)
            max_probs[:, : (i+1)] = -1.0
            _, idx = torch.topk(max_probs, k=args.topk, dim=-1)

            # 根据 idx 取标签
            z_labels = torch.gather(z_indices, dim=1, index=idx)

            # 根据 idx 取 logits 分布
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))
            pred_logits = torch.gather(logits, dim=1, index=idx_expanded)



            loss = F.cross_entropy(pred_logits.reshape(args.global_batch_size * args.topk, -1), z_labels.reshape(args.global_batch_size * args.topk))
            # loss = F.cross_entropy(
            #     logits[:,i:,:].reshape(-1, logits.size(-1)),
            #     z_indices[:,i:].reshape(-1)
            # )
            
            
            # yuzhehao写法
            # topk_vals, topk_idx = torch.topk(max_probs, k=args.topk, dim=-1)
            # conf_mask = torch.zeros_like(max_probs)
            # conf_mask.scatter_(1, topk_idx, 1.0)

            # valid_logits = logits.reshape(-1, logits.size(-1))
            # valid_targets = z_indices.reshape(-1)
            # valid_mask = conf_mask.reshape(-1)


            # loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
            # loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)  

            """
            
            conf = probs.max(dim=-1).values                      # [B, T]
            T = conf.size(1)
            prefix_mask = torch.arange(T, device=conf.device).unsqueeze(0) <= i  # 屏蔽已生成位置
            conf_masked = conf.masked_fill(prefix_mask, float('-inf'))

            # 有效的 k（避免剩余长度 < k 的问题）
            k_eff = max(1, min(args.topk, T - (i + 1)))

            # 前向仍然用硬 top-k（不求导）
            idx = torch.topk(conf_masked.detach(), k=k_eff, dim=-1).indices  # detach 避免在选择上建图
            hard_mask = torch.zeros_like(conf, dtype=logits.dtype)
            hard_mask.scatter_(1, idx, 1.0)                                  # [B, T]，仅未生成区间被选为1

            # 反向用软权（可导），直通估计器：前向=hard，反向=soft
            tau = 0.5  # 可调温度（0.3~1 之间调）
            soft_w = torch.softmax(conf_masked / tau, dim=-1)                # [B, T]，前缀自然为0
            st_mask = (hard_mask - soft_w).detach() + soft_w                 # ST trick

            # ========= 计算 loss（与原逻辑等价，仅把 valid_mask 换成 st_mask） =========
            valid_logits  = logits.reshape(-1, logits.size(-1))
            valid_targets = z_indices.reshape(-1)
            loss_vec = F.cross_entropy(valid_logits, valid_targets, reduction="none").reshape_as(conf)

            loss = (loss_vec * st_mask).sum() / (st_mask.sum() + 1e-6)

            """
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()

            
            for n,p in tpn_model.named_parameters():
                if p.grad is not None:
                    print(n, p.grad.norm().item())
                    #break
                else:
                    print("NO GRADS found on tpn_model params!")
            
            #if args.max_grad_norm != 0.0:
            #    scaler.unscale_(optimizer)
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    tpn_model_weight = tpn_model.state_dict()  
                    checkpoint = {
                        "model": tpn_model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    tpn_model.eval()
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, default="/mnt/disk0/liuxh/LlamaGen/imagenet_code_c2i_flip_ten_crop")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default="/mnt/disk0/liuxh/models/c2i_B_256.pt", help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results_tpn_multicross")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-1, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--freq_probs_path", type=str, default='prior_counts.npy') 
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--rn", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    main(args)
