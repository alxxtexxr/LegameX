import os
import json

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering
from peft import LoraConfig

from .LegamexLora import LegamexLora


class LegamexLoraModel(nn.Module):
    def __init__(
            self, base_model, task_type, 
            
            # Reference (ref) and transfer (tfr) LoRA parameters
            ref_lora_config, tfr_lora_config, 
            
            # Gate parameters
            gate_warmup_steps, gate_scheduler_type='cosine', 
            gate_rank=1, gate_alpha=1, gate_dropout=0.0, 
            gate_bias=False, gate_use_rslora=False
        ):
        super().__init__()
        
        # Determine the base model name or path
        self.base_model_name_or_path = (
            ref_lora_config.base_model_name_or_path
            if hasattr(ref_lora_config, 'base_model_name_or_path')
            else base_model.config.name_or_path
        )
        if base_model.config.name_or_path != self.base_model_name_or_path:
            print("Using the base model name from the reference LoRA:", self.base_model_name_or_path)
        
        # Store the parameters
        self.base_model = base_model
        self.task_type = task_type
        self.ref_lora_config = ref_lora_config
        self.tfr_lora_config = tfr_lora_config
        self.gate_warmup_steps = gate_warmup_steps
        self.gate_scheduler_type = gate_scheduler_type
        self.gate_rank = gate_rank
        self.gate_alpha = gate_alpha
        self.gate_dropout = gate_dropout
        self.gate_bias = gate_bias
        self.gate_use_rslora = gate_use_rslora

        # Wrap the target modules with LegameX
        self.legamex_modules = {}
        self._wrap_target_modules_with_legamex()


    def _get_parent_module(self, module_name):
            parts = module_name.split('.')
            parent_module = self.base_model
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            return parent_module, parts[-1]


    def _wrap_target_modules_with_legamex(self):
        assert self.ref_lora_config.target_modules == self.tfr_lora_config.target_modules, \
            "Target modules for reference and transfer LoRA must be the same."
        target_modules = self.tfr_lora_config.target_modules
        
        for module_name, module in self.base_model.named_modules():
            # Skip the modules that are already wrapped with LegamexLora
            if isinstance(module, LegamexLora):
                module_name = module_name.replace('.', '__DOT__')
                self.legamex_modules[module_name] = module
                continue
            
            if any(target_module in module_name for target_module in target_modules):
                # Wrap the module with LegamexLora
                parent_module, child_name = self._get_parent_module(module_name)
                legamex_module = LegamexLora(
                    base_module=module,
                    ref_lora_config=self.ref_lora_config,
                    tfr_lora_config=self.tfr_lora_config,
                    gate_warmup_steps=self.gate_warmup_steps,
                    gate_scheduler_type=self.gate_scheduler_type,
                    gate_rank=self.gate_rank,
                    gate_alpha=self.gate_alpha,
                    gate_dropout=self.gate_dropout,
                    gate_bias=self.gate_bias,
                    gate_use_rslora=self.gate_use_rslora,
                )
                setattr(parent_module, child_name, legamex_module)
                
                # Store the wrapped module
                module_name = module_name.replace('.', '__DOT__')
                self.legamex_modules[module_name] = legamex_module
                
            # Store the qa_outputs modules if task_type is 'QUESTION_ANS'
            if isinstance(module, nn.Linear) and 'qa_outputs' in module_name and self.task_type == 'QUESTION_ANS':
                module_name = module_name.replace('.', '__DOT__')
                self.legamex_modules[module_name] = module
                
        # Freeze all parameters in the base model and unfreeze only the LegameX modules except for the reference LoRA
        self.freeze_all(verbose=False)
        self.unfreeze_legamex_except_ref_lora(verbose=True)


    def _json_safe(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        return obj


    def freeze_all(self, verbose=False):
        for p in self.base_model.parameters():
            p.requires_grad = False
        if verbose:
            print("All modules are frozen.")


    def unfreeze_legamex_except_ref_lora(self, verbose=False):
        for module_name, module in self.legamex_modules.items():
            if isinstance(module, LegamexLora):
                # Unfreeze the transfer LoRA weights and biases
                module.lora.tfr.A.weight.requires_grad = True # type: ignore
                module.lora.tfr.B.weight.requires_grad = True # type: ignore
                if module.lora.tfr.use_bias: # type: ignore
                    if hasattr(module.lora.tfr.A, 'bias') and module.lora.tfr.A.bias is not None: # type: ignore
                        module.lora.tfr.A.bias.requires_grad = True # type: ignore
                    if hasattr(module.lora.tfr.B, 'bias') and module.lora.tfr.B.bias is not None: # type: ignore
                        module.lora.tfr.B.bias.requires_grad = True # type: ignore
                
                # Unfreeze the gate weights and biases
                module.gate.A.weight.requires_grad = True # type: ignore
                module.gate.B.weight.requires_grad = True # type: ignore
                if module.gate.use_bias:
                    if hasattr(module.gate.A, 'bias') and module.gate.A.bias is not None: # type: ignore
                        module.gate.A.bias.requires_grad = True # type: ignore
                    if hasattr(module.gate.B, 'bias') and module.gate.B.bias is not None: # type: ignore
                        module.gate.B.bias.requires_grad = True # type: ignore

            # Unfreeze the qa_outputs weight and bias if task_type is 'QUESTION_ANS'
            elif isinstance(module, nn.Linear) and 'qa_outputs' in module_name and self.task_type == 'QUESTION_ANS':    
                module.weight.requires_grad = True
                if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True

        if verbose:
            trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.base_model.parameters())
            print(f"Trainable parameters: {trainable:,} / {total:,}")


    def load_lora_weights(self, lora_path, lora_type):
        assert lora_type in ['ref', 'tfr'], "Invalid LoRA type. Must be 'ref' or 'tfr'."
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")
        
        if lora_path.endswith('.safetensors'):
            state_dict = load_file(lora_path)
        else:
            state_dict = torch.load(lora_path, map_location='cpu')
            
        processed_module_names = set()
        for key in state_dict.keys():
            key_prefix, key_rest = key.rsplit('model.', 1)
            module_name = key_rest.split('.lora', 1)[0]
            module_name_sanitized = module_name.replace('.', '__DOT__')
            
            if module_name_sanitized in processed_module_names:
                continue
            
            if module_name_sanitized in self.legamex_modules:
                prefix = key_prefix + 'model.' + module_name
                self.legamex_modules[module_name_sanitized].load_lora_weights(state_dict, prefix, lora_type)
            else:
                print("LoRA weights not found for module:", module_name)
                
            processed_module_names.add(module_name_sanitized)

        print(f"LoRA weights loaded from {lora_path} into model.")


    @classmethod
    def from_pretrained(cls, save_dir, **kwargs):
        # Load the LegameX configuration
        with open(os.path.join(save_dir, 'legamex_config.json'), 'r') as f:
            config = json.load(f)
            
        # Re-create the base model
        base_model_id = config['base_model_name_or_path']
        task_type = config['task_type']
        if task_type == 'QUESTION_ANS':
            base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_id)
        elif task_type == 'FEATURE_EXTRACTION':
            base_model = AutoModelForMaskedLM.from_pretrained(base_model_id)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Load the reference and transfer LoRA configurations
        ref_lora_config = LoraConfig(**config['ref_lora_config'])
        tfr_lora_config = LoraConfig(**config['tfr_lora_config'])
        
        # Re-create the LegameX model
        model = cls(
            base_model=base_model,
            task_type=task_type,
            ref_lora_config=ref_lora_config,
            tfr_lora_config=tfr_lora_config,
            gate_warmup_steps=config['gate_warmup_steps'],
            gate_scheduler_type=config['gate_scheduler_type'],
            gate_rank=config['gate_rank'],
            gate_alpha=config['gate_alpha'],
            gate_dropout=config['gate_dropout'],
            gate_bias=config['gate_bias'],
            gate_use_rslora=config['gate_use_rslora'],
        )
        
        # Load the LegameX state dict
        legamex_state_dict = torch.load(os.path.join(save_dir, 'legamex.pt'), map_location='cpu')
        for k, v in legamex_state_dict.items():
            parts = k.split('.', 1) # ['module_name', '<rest_of_path>']
            if len(parts) == 2:
                module_name, param_name = parts
                module = model.legamex_modules.get(module_name)
                if module is not None:
                    param = dict(module.named_parameters()).get(param_name)
                    if param is not None:
                        param.data.copy_(v)
        
        return model


    def save_pretrained(self, save_dir):
        # Create save directories
        os.makedirs(save_dir, exist_ok=True)
        
        tfr_dir = os.path.join(save_dir, 'tfr')
        os.makedirs(tfr_dir, exist_ok=True)
        
        # Store the LegameX and transfer component state dicts
        state_dict = {}
        tfr_state_dict = {}
        
        for module_name, module in self.legamex_modules.items():
            for param_name, param in module.named_parameters():
                # Skip the base_module parameters
                if 'base_module' in param_name:
                    continue
                
                # Store the rest of parameters in the LegameX state dict
                full_param_name = f'{module_name}.{param_name}'
                param_cpu = param.detach().cpu()
                state_dict[full_param_name] = param_cpu
                
                # Store the transfer component and qa_outputs parameters separately
                if 'lora.tfr' in param_name or 'qa_outputs' in module_name:
                    full_param_name_sanitized = full_param_name.replace('__DOT__', '.').replace('lora.tfr.', 'lora_')
                    tfr_param_cpu = param_cpu

                    # If the parameter is a transfer component weight, apply the gate complement weight to it
                    if 'lora.tfr' in param_name and 'weight' in param_name:
                        gate_weight_name = param_name.replace('lora.tfr', 'gate')
                        
                        try:
                            gate_param = module.get_parameter(gate_weight_name)
                            gate_weight = gate_param.detach().cpu()
                            gate_comp_weight = 1.0 - gate_weight
                            tfr_param_cpu = tfr_param_cpu * gate_comp_weight
                        except AttributeError:
                            raise AttributeError(
                                f"Could not find matching gate weight component '{gate_weight_name}' "
                                f"in module '{module_name}' to pair with transfer component weights."
                            )
                        
                    tfr_state_dict[full_param_name_sanitized] = tfr_param_cpu
                    
        torch.save(state_dict, os.path.join(save_dir, 'legamex.pt'))
        
        # Save the transfer component state dict and its configuration
        save_file(tfr_state_dict, os.path.join(tfr_dir, 'adapter_model.safetensors'))
        with open(os.path.join(tfr_dir, 'adapter_config.json'), 'w') as f:
            json.dump(self._json_safe(self.tfr_lora_config.to_dict()), f, indent=4)
        
        # Save the LegameX configuration for rebuilding wrapper
        config = {
            'base_model_name_or_path': self.base_model_name_or_path,
            'task_type': self.task_type,
            'ref_lora_config': self._json_safe(self.ref_lora_config.to_dict()),
            'tfr_lora_config': self._json_safe(self.tfr_lora_config.to_dict()),
            'gate_warmup_steps': self.gate_warmup_steps,
            'gate_scheduler_type': self.gate_scheduler_type,
            'gate_rank': self.gate_rank,
            'gate_alpha': self.gate_alpha,
            'gate_dropout': self.gate_dropout,
            'gate_bias': self.gate_bias,
            'gate_use_rslora': self.gate_use_rslora,
        }
        with open(os.path.join(save_dir, 'legamex_config.json'), 'w') as f:
            json.dump(config, f, indent=4)


    def forward(self, *args, **kwargs):
            kwargs.pop('num_items_in_batch', None)
            return self.base_model.forward(*args, **kwargs)


    def set_training_step(self, training_step):
        for module in self.legamex_modules.values():
            if hasattr(module, 'set_training_step'):
                module.set_training_step(training_step)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
