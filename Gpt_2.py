from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import math
import inspect
import time
import os
import numpy as np

    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, nT, C = x.shape
        x = self.c_attn(x)
        q, k, v = x.split(self.n_embd, dim = 2)

        q = q.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
        attn = torch.masked_fill(attn, self.bias[:, :, :nT, :nT] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)


        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, nT, C)
        x = self.c_proj(x)
        return x
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
    
        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal(module.weight, mean = 0.0, std = 0.02)
                
    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type == "gpt2"
        from transformers import GPT2LMHeadModel
        print("loads weights  from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embd = 768)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"{k} shape mismatch"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"{k} shape mismatch"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        "this function defines which kind of parameters are doing weight decay and which are not"
        #basic we dont perform weight decay on 1d tensors like nn.linear, and do it on other tensors
        #start with all of the candidate parameters which required grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        #i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms dont
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number decayed parameter tensors:  {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non_decayed parameter tensor: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        #create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas= (0.9, 0.95), eps = 1e-8)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt_convert= npt.astype(np.int64)
    ptt = torch.tensor(npt_convert, dtype=torch.long)
    return ptt

class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T

        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "/Users/eddy/Documents/Build_from_scratch/edu_fineweb10BT"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.processes
        return x, y

#--------------------------------------------------------------------------------
#inference
    
#num_return_sequences = 5
#max_length = 30
#model = GPT.from_pretrained('gpt2')
#
#model.eval()
#
#import tiktoken
#enc = tiktoken.get_encoding("gpt2")
#tokens = enc.encode("Hello, I'm a language model,")
#tokens = torch.tensor(tokens, dtype = torch.long)
#x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#
#torch.manual_seed(44)
#
#while x.size(1) < max_length:
#    with torch.no_grad():
#        logits = model(x)
#        logits = logits[:, -1, :]
#        probs = F.softmax(logits, dim = -1)
#        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
#        ix = torch.multinomial(topk_probs, 1)
#        xcol = torch.gather(topk_indices, -1, ix)
#        x = torch.cat((x, xcol), dim = 1)
#
#
#for i in range(num_return_sequences):
#    tokens = x[i, :max_length].tolist()
#    decoded = enc.decode(tokens)
#    print(">", decoded)


#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#training
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda: {ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0          #this process will do logging, checking etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"----using device----: {device}")

import tiktoken
B = 4
T = 512

total_batch_size = 524488   #not any more
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size; {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataloaderLite(B, T, process_rank = ddp_rank, num_processes = ddp_world_size, split = 'train')
val_loader = DataloaderLite(B, T, process_rank = ddp_rank, num_processes = ddp_world_size, split = 'val')


model = GPT(GPTConfig())
model.to(device)
#model = torch.compile(model)           #mps not supported
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_step = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) /warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_step:
        return min_lr
    decay_ratio = (it - warmup_steps)/(max_step - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device=device)
for step in range(max_step):
    t0 = time.time()
    #evaluate model once in a while
    if step%100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss:{val_loss_accum.item():.4f}")

        
    #training loop
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()               #make sure clean gradient first
    logits, loss = model(x, y)          #compute loss and logits
    if ddp:
        model.require_backward_grad_sync = (step == grad_accum_steps - 1)
    loss.backward()                     #backward the loss
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()                    #compute the new parameters
    #torch.cuda.synchronize()            #wait for the GPU to finish work
    torch.mps.synchronize()             #mps for macbook
    t1 = time.time()
    dt = t1 - t0
    if master_process:
        print(f"step {step}, loss: {loss.item()} | dt: {dt * 1000} ms")
    
if ddp:
    destroy_process_group()