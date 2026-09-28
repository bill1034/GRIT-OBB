"""State-dict checkpoints; no whole-model pickle and no network on restore."""
from __future__ import annotations
from pathlib import Path
import os
import torch
from .model import build_model


def save_checkpoint(path,model,optimizer=None,step: int=0,extra: dict | None=None) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state={'format_version':1,'config':model.config.to_dict(),'model':model.state_dict(),
           'optimizer':None if optimizer is None else optimizer.state_dict(),'step':int(step),
           'has_relation_module':model.relation is not None,
           'pretrained_status':dict(model.pretrained_status),'torch_version':str(torch.__version__),
           'rng_state':torch.get_rng_state(),'extra':{} if extra is None else extra}
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def load_checkpoint(path,model=None,optimizer=None,map_location='cpu',restore_rng: bool=False):
    checkpoint=torch.load(path,map_location=map_location,weights_only=True)
    if checkpoint.get('format_version')!=1:
        raise ValueError('Unsupported checkpoint format')
    if model is None:
        if optimizer is not None:
            raise ValueError('To restore optimizer, provide the model associated with that optimizer')
        config=dict(checkpoint['config']); config['weights']=None
        config['use_relation']=checkpoint.get('has_relation_module',config['use_relation'])
        model=build_model(config).to(map_location)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.set_ablation(checkpoint['config']['use_relation'],checkpoint['config']['use_iou_loss'])
    model.config.weights=checkpoint['config']['weights']
    # Restore original metadata, but never download ImageNet on checkpoint load.
    model.pretrained_status=dict(checkpoint['pretrained_status'])
    if optimizer is not None:
        if checkpoint['optimizer'] is None:
            raise ValueError('Checkpoint has no optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer'])
    if restore_rng:
        torch.set_rng_state(checkpoint['rng_state'].cpu())
    metadata={k:v for k,v in checkpoint.items() if k not in ('model','optimizer','rng_state')}
    return model,metadata
