from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import json
from pathlib import Path


@dataclass
class ModelConfig:
    num_classes: int = 15
    weights: str | None = 'IMAGENET1K_V1'
    trainable_backbone_layers: int = 3
    input_size: int = 1024  # Documentation/default profiling size; never silently resizes images.
    pool_size: int = 7
    ps_channels: int = 10
    sampling_ratio: int = 2
    head_dim: int = 1024
    use_relation: bool = True
    relation_dim: int = 128
    relation_heads: int = 4
    relation_gate_init: float = .1
    use_iou_loss: bool = True
    iou_weight: float = 2.
    context_long: float = 1.2
    context_short: float = 1.4
    rpn_pre_nms_train: int = 6000
    rpn_pre_nms_test: int = 6000
    rpn_post_nms_train: int = 1000
    rpn_post_nms_test: int = 1000
    rpn_nms_threshold: float = .7
    roi_batch_size: int = 512
    roi_positive_fraction: float = .25
    roi_positive_iou: float = .5
    score_threshold: float = .05
    final_nms_threshold: float = .5
    pre_nms_topk: int = 5000
    max_detections: int = 1000

    def validate(self) -> None:
        if self.num_classes<1 or self.head_dim<4 or self.input_size<32:
            raise ValueError('Invalid class count, head dimension or input size')
        if self.weights not in (None,'IMAGENET1K_V1','IMAGENET1K_V2'):
            raise ValueError('weights must be None, IMAGENET1K_V1 or IMAGENET1K_V2')
        if not 0<=self.trainable_backbone_layers<=5:
            raise ValueError('trainable_backbone_layers must be in [0,5]')
        for key in ('pool_size','ps_channels','sampling_ratio','relation_dim','relation_heads',
                    'roi_batch_size','rpn_pre_nms_train','rpn_pre_nms_test',
                    'rpn_post_nms_train','rpn_post_nms_test','pre_nms_topk','max_detections'):
            if getattr(self,key)<1:
                raise ValueError(f'{key} must be positive')
        if self.relation_dim%self.relation_heads or self.relation_dim%4:
            raise ValueError('relation_dim must be divisible by heads and by four')
        for key in ('rpn_nms_threshold','roi_positive_iou','score_threshold','final_nms_threshold'):
            if not 0<=getattr(self,key)<=1:
                raise ValueError(f'{key} must be in [0,1]')
        if not 0<self.roi_positive_fraction<=1 or min(self.context_long,self.context_short)<=0 or self.iou_weight<0:
            raise ValueError('Invalid sampling fraction, context or loss weight')
        if not -1<self.relation_gate_init<1:
            raise ValueError('Gate initialization must lie strictly between -1 and 1')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_value(cls, value: 'ModelConfig | dict | str | Path') -> 'ModelConfig':
        if isinstance(value,cls):
            result=cls(**asdict(value))
        else:
            data=json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else dict(value)
            unknown=set(data)-{f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
            result=cls(**data)
        result.validate()
        return result
