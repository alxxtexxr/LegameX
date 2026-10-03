import math

import torch
import torch.nn as nn


class LegamexLora(nn.Module):
    def __init__(
            self, base_module, 
            
            # Reference (ref) and transfer (tfr) LoRA parameters
            ref_lora_config, tfr_lora_config,
                    
            # Gate parameters
            gate_warmup_steps, gate_scheduler_type='cosine', 
            gate_rank=1, gate_alpha=1, gate_dropout=0.0, 
            gate_bias=False, gate_use_rslora=False,
        ):
        super().__init__()
        
        assert gate_scheduler_type in ['linear', 'cosine', 'quadratic', 'cubic', 'sqrt'], (
            f"Unknown gate scheduler type: {gate_scheduler_type}. "
            "Must be one of ['linear', 'cosine', 'quadratic', 'cubic', 'sqrt']."
        )
        
        self.base_module = base_module
        self.gate_warmup_steps = gate_warmup_steps
        self.gate_scheduler_type = gate_scheduler_type
        self.training_step = 0
        
        # Determine in_features and out_features from the base module
        in_features = getattr(base_module, 'in_features', None)
        out_features = getattr(base_module, 'out_features', None)
        if in_features is None or out_features is None:
            raise ValueError(f"Cannot determine in_features or out_features from {base_module}.")
        
        # Initialize reference (ref) and transfer (tfr) LoRA modules
        self.lora = nn.ModuleDict({
            'ref': self._init_lora_module(in_features, out_features, 
                                          rank=ref_lora_config.r, 
                                          alpha=ref_lora_config.lora_alpha, 
                                          dropout=ref_lora_config.lora_dropout, 
                                          bias=False if ref_lora_config.bias == 'none' else True, 
                                          use_rslora=ref_lora_config.use_rslora),
            'tfr': self._init_lora_module(in_features, out_features, 
                                          rank=tfr_lora_config.r, 
                                          alpha=tfr_lora_config.lora_alpha, 
                                          dropout=tfr_lora_config.lora_dropout, 
                                          bias=False if tfr_lora_config.bias == 'none' else True, 
                                          use_rslora=tfr_lora_config.use_rslora),
        })
        
        # Initialize gate modules
        self.gate = self._init_lora_module(in_features, out_features, 
                                           rank=gate_rank, alpha=gate_alpha, 
                                           dropout=gate_dropout, bias=gate_bias, 
                                           use_rslora=gate_use_rslora)
        self.gate_act = nn.Sigmoid()


    def _init_lora_module(self, in_features, out_features, rank, alpha, dropout, bias, use_rslora):
        device = self.base_module.weight.device
        dropout_module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        
        # Initialize LoRA modules with extracted parameters
        lora_A = nn.Linear(in_features, rank, bias=bias, device=device)
        lora_B = nn.Linear(rank, out_features, bias=bias, device=device)
        
        # Initialize weights for the LoRA modules
        # nn.init.normal_(lora_A.weight, mean=0.0, std=1 / math.sqrt(rank))
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_B.weight)
        
        lora_module = nn.ModuleDict({
            'A': lora_A,
            'B': lora_B,
            'dropout': dropout_module,
        })
        lora_module.scaling = scaling
        lora_module.use_bias = bias
        return lora_module


    def load_lora_weights(self, state_dict, prefix, lora_type):
        assert lora_type in ['ref', 'tfr'], "Invalid LoRA type. Must be 'ref' or 'tfr'."
        
        device_A = self.lora[lora_type].A.weight.device # type: ignore
        device_B = self.lora[lora_type].B.weight.device # type: ignore

        key_A = f'{prefix}.lora_A.weight'
        key_B = f'{prefix}.lora_B.weight'

        with torch.no_grad():
            self.lora[lora_type].A.weight.copy_(state_dict[key_A].to(device_A)) # type: ignore
            self.lora[lora_type].B.weight.copy_(state_dict[key_B].to(device_B)) # type: ignore

            if self.lora[lora_type].use_bias:
                if f'{prefix}.lora_A.bias' in state_dict:
                    self.lora[lora_type].A.bias.copy_(state_dict[f'{prefix}.lora_A.bias'].to(device_A)) # type: ignore
                if f'{prefix}.lora_B.bias' in state_dict:
                    self.lora[lora_type].B.bias.copy_(state_dict[f'{prefix}.lora_B.bias'].to(device_B)) # type: ignore    


    def gate_scheduler(self):
        start = 0.0
        end = 1.0
        step = self.training_step
        warmup_steps = self.gate_warmup_steps
        scheduler_type = self.gate_scheduler_type
        
        if warmup_steps <= 0:
            return end
        
        t = min(max(step / warmup_steps, 0.0), 1.0)
        
        if scheduler_type == 'linear':
            p = t
        elif scheduler_type == 'cosine':
            # cosine warmup
            p = 0.5 * (1.0 - math.cos(math.pi * t))
        elif scheduler_type == 'quadratic':
            p = t**2
        elif scheduler_type == 'cubic':
            p = t**3
        elif scheduler_type == 'sqrt':
            p = math.sqrt(t)
        else:
            raise ValueError(
                f"Unknown gate scheduler type: {scheduler_type}. "
                "Must be one of ['linear', 'cosine', 'quadratic', 'cubic', 'sqrt']."
            )
        
        return start + (end - start) * p


    def forward(self, x):
        # Compute the base layer output
        base_out = self.base_module(x)
        
        # Check if the input tensor needs to be converted to LoRA module's dtype
        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            # TODO: Add assertion to check if all weights have same dtype
            x = x.to(self.lora.tfr.A.weight.dtype) # pyright: ignore
        
        # Compute the outputs for the reference and transfer LoRA
        ref_lora_out = self.lora.ref.B(self.lora.ref.dropout(self.lora.ref.A(x))) * self.lora.ref.scaling # type: ignore
        tfr_lora_out = self.lora.tfr.B(self.lora.tfr.dropout(self.lora.tfr.A(x))) * self.lora.tfr.scaling # type: ignore
        
        # Compute the gate output and its complement
        gate_out = self.gate_act(self.gate.B(self.gate.dropout(self.gate.A(x)))) # type: ignore
        gate_out = gate_out * self.gate_scheduler() # Schedule gate output based on training step and warmup steps
        gate_comp_out = torch.ones(gate_out.shape, dtype=gate_out.dtype, device=gate_out.device) - gate_out
        
        # Compute the LoRA output
        lora_out = gate_out * ref_lora_out + gate_comp_out * tfr_lora_out
        
        # Check if the LoRA output needs to be converted back to the base layer's dtype
        if requires_conversion:
            lora_out = lora_out.to(base_out.dtype)
            
        return base_out + lora_out


    def set_training_step(self, training_step):
        self.training_step = training_step
