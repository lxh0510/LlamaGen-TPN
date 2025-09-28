# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ , hidden_states = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        hidden_states_combined = hidden_states
        cond_hidden_states, uncond_hidden_states = torch.split(hidden_states_combined, len(hidden_states_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            hidden_states = uncond_hidden_states + (cond_hidden_states - uncond_hidden_states) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ , hidden_states = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)[0], sample(logits, **sampling_kwargs)[1], hidden_states


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob, hidden_states = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

def find_addition_tokens(tpn_model, hidden_states, generated_labels, embedding_layer, prior_embeddings, sample_entropy = True, sample_logits = True):
    hidden_states = torch.cat(hidden_states, dim=0).to(tpn_model.device).permute(1, 0, 2)
    generated_labels = torch.cat(generated_labels, dim=1).to(hidden_states.device)
    generated_embeddings = embedding_layer(generated_labels)
    ungenerated_embeddings  = prior_embeddings[:,generated_labels.shape[1]:,:]
    #embeddings = torch.cat([generated_embeddings, ungenerated_embeddings], dim=1)
    labels = torch.zeros((generated_embeddings.shape[0], prior_embeddings.shape[1])).to(hidden_states.device)
    labels[:,:generated_labels.shape[1]] = generated_labels
    labels[:,generated_labels.shape[1]:] = embedding_layer.weight.shape[0]
    logits = tpn_model(generated_embeddings, ungenerated_embeddings, hidden_states, labels)
    seq_len = logits.shape[1]
    logits = torch.softmax(logits, dim=-1)

    # 计算熵不如计算置信度
    #entropy = -(logits * torch.log(logits + 1e-6)).sum(dim=2)
    #entropy[:,:generated_labels.shape[1]] = -1e9
    #entropy = torch.softmax(entropy, dim=-1)

    # 使用置信度作为衡量标准
    max_probs, _ = logits.max(dim=-1)
    max_probs[:, : generated_labels.shape[1]] = -1e9


    if sample_entropy:
        weights = torch.clamp_min(max_probs, 0)
        idx = torch.multinomial(weights, num_samples=1)
    else:
        _, idx = torch.topk(max_probs, k=1, dim=-1)
    
    selected_logits = torch.gather(logits, 1, idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))).squeeze(1)
    if sample_logits:
        token_idx = torch.multinomial(selected_logits, num_samples=1)
    else:
        _, token_idx = torch.topk(selected_logits, k=1, dim=-1)

    return idx, token_idx


def decode_n_tokens_tpn(
    model, tpn_model, prior_probs, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, start_token: int,
    **sampling_kwargs):
    bs = cur_token.shape[0]
    seq_len = prior_probs.shape[1]
    vocab_size = prior_probs.shape[2]
    new_tokens, new_probs = [], []
    all_tokens = torch.ones(bs, seq_len) * vocab_size
    all_tokens = all_tokens.long().to(cur_token.device)
    cfg_flag = True
    embedding_layer = model.tok_embeddings
    all_token_embeddings = embedding_layer.weight
    prior_embeddings = torch.matmul(prior_probs, all_token_embeddings)
    all_hidden_states = []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob, hidden_states = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )

            # to do: 验证步
            all_hidden_states.append(hidden_states)

            input_pos += 1
            new_tokens.append(next_token.clone())
            all_tokens[:,i] = next_token.squeeze(-1).clone()
            new_probs.append(next_prob.clone())

            if input_pos >= start_token: 

                addition_token_loc, addition_token_idx = find_addition_tokens(tpn_model, all_hidden_states, new_tokens, embedding_layer, prior_embeddings)

                # 这么写有问题
                #all_tokens[:,addition_token_loc.squeeze(0)] = addition_token_idx
                rows = torch.arange(bs, device=all_tokens.device)
                cols = addition_token_loc.squeeze(1)
                all_tokens[rows, cols] = addition_token_idx.squeeze(1)
                finished = torch.all(all_tokens != vocab_size)

                if finished:
                    print("llm 生成的token : " , input_pos)
                    return  all_tokens.tolist(), new_probs
              
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

@torch.no_grad()
def generate(model, tpn_model, prior_probs,  cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    if tpn_model is not None:
        generated_tokens, _ = decode_n_tokens_tpn(model, tpn_model, prior_probs, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, start_token=200, **sampling_kwargs)
        seq[:, T+1:] = torch.tensor(generated_tokens, device=device)[:,:-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
        seq[:, T+1:] = torch.cat(generated_tokens, dim = 1)

    return seq[:, T:]
