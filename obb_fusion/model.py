from __future__ import annotations
from collections import OrderedDict
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import FrozenBatchNorm2d, box_iou
from .config import ModelConfig
from .geometry import (validate_rboxes, canonicalize, rbox_to_hbb, hbb_to_rbox,
    decode_boxes, equivalent_code_loss, aligned_rotated_iou, pairwise_rotated_iou)
from .sampling import MultiScaleRPSAlign
from .relation import GaussianRelationResidual
from .postprocess import postprocess


@torch.no_grad()
def match_labels(proposals: Tensor, gt_boxes: Tensor, gt_labels: Tensor,
                 positive_iou: float, rotated: bool) -> tuple[Tensor,Tensor]:

    if len(gt_boxes)==0:
        return (torch.zeros(len(proposals),device=proposals.device,dtype=torch.long),
                torch.zeros(len(proposals),device=proposals.device,dtype=torch.long))
    overlaps=pairwise_rotated_iou(proposals,gt_boxes) if rotated else box_iou(proposals,gt_boxes)
    best,matched=overlaps.max(1)
    labels=gt_labels[matched]+1
    labels=torch.where(best>=positive_iou,labels,torch.zeros_like(labels))
    return labels,matched


@torch.no_grad()
def balanced_indices(labels: Tensor, batch_size: int, fraction: float) -> tuple[Tensor,Tensor]:
    positive=torch.where(labels>0)[0]
    negative=torch.where(labels==0)[0]
    npos=min(len(positive),max(1,int(batch_size*fraction)))
    nneg=min(len(negative),batch_size-npos)
    pos=positive[torch.randperm(len(positive),device=labels.device)[:npos]]
    neg=negative[torch.randperm(len(negative),device=labels.device)[:nneg]]
    return pos,torch.cat((pos,neg))


class GeometryRelationRoITransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config=ModelConfig.from_value(config)
        cfg=self.config
        weight_enum=None if cfg.weights is None else ResNet50_Weights[cfg.weights]


        self.backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=weight_enum,
            norm_layer=FrozenBatchNorm2d,trainable_layers=cfg.trainable_backbone_layers)
        self.pretrained_status={'requested':cfg.weights,'loaded':weight_enum is not None,
            'source':None if weight_enum is None else weight_enum.url,
            'scope':'ResNet-50 only; FPN and detection heads are newly initialized'}


        anchors=AnchorGenerator(sizes=((16,32),(32,64),(64,128),(128,256),(256,512)),
            aspect_ratios=((.5,1.,2.),)*5)
        self.rpn=RegionProposalNetwork(anchors,RPNHead(256,6),fg_iou_thresh=.7,
            bg_iou_thresh=.3,batch_size_per_image=256,positive_fraction=.5,
            pre_nms_top_n={'training':cfg.rpn_pre_nms_train,'testing':cfg.rpn_pre_nms_test},
            post_nms_top_n={'training':cfg.rpn_post_nms_train,'testing':cfg.rpn_post_nms_test},
            nms_thresh=cfg.rpn_nms_threshold,score_thresh=0.)
        thin_channels=cfg.pool_size**2*cfg.ps_channels
        self.thin=nn.Conv2d(256,thin_channels,1)
        self.align=MultiScaleRPSAlign(cfg.pool_size,cfg.ps_channels,cfg.sampling_ratio)
        self.roi_learner=nn.Linear(thin_channels,5)
        self.roi_head=nn.Sequential(nn.Linear(thin_channels,cfg.head_dim),
                                   nn.LayerNorm(cfg.head_dim),nn.ReLU())
        self.classifier=nn.Linear(cfg.head_dim,cfg.num_classes+1)
        self.regressor=nn.Linear(cfg.head_dim,5)
        nn.init.xavier_uniform_(self.thin.weight); nn.init.zeros_(self.thin.bias)
        nn.init.normal_(self.roi_learner.weight,std=.001); nn.init.zeros_(self.roi_learner.bias)
        nn.init.xavier_uniform_(self.roi_head[0].weight); nn.init.zeros_(self.roi_head[0].bias)
        nn.init.normal_(self.classifier.weight,std=.01); nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.regressor.weight,std=.001); nn.init.zeros_(self.regressor.bias)
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406])[:,None,None])
        self.register_buffer('image_std',torch.tensor([.229,.224,.225])[:,None,None])


        self.relation=(GaussianRelationResidual(cfg.head_dim,cfg.relation_dim,
            cfg.relation_heads,cfg.relation_gate_init) if cfg.use_relation else None)

    def set_ablation(self, relation: bool, iou: bool) -> None:

        if relation and self.relation is None:
            raise ValueError('This base instance has no relation module; build a full instance to enable it')
        self.config.use_relation=bool(relation)
        self.config.use_iou_loss=bool(iou)

    def _prepare_images(self, images: Tensor | Sequence[Tensor]) -> ImageList:
        if isinstance(images,Tensor):
            if images.ndim==3:
                images=[images]
            elif images.ndim==4:
                images=list(images.unbind(0))
            else:
                raise ValueError('images must be [B,C,H,W], [C,H,W], or a list of [C,H,W]')
        else:
            images=list(images)
        if not images:
            raise ValueError('Empty image batches are not supported')
        sizes=[]; normalized=[]
        for image in images:
            if image.ndim!=3 or image.shape[0] not in (1,3) or not image.is_floating_point():
                raise ValueError('Each image must be floating point [1 or 3,H,W] in [0,1]')
            if image.device!=self.image_mean.device:
                raise ValueError('Move images and model to the same device before calling the detector')
            if min(image.shape[-2:])<16 or not torch.isfinite(image).all():
                raise ValueError('Invalid spatial size or nonfinite image pixels')
            if image.min() < -1e-6 or image.max() > 1+1e-6:
                raise ValueError('Expected image pixels in [0,1], not uint8 range or pre-normalized input')
            sizes.append((image.shape[-2],image.shape[-1]))
            rgb=image.float().expand(3,-1,-1) if image.shape[0]==1 else image.float()
            normalized.append((rgb-self.image_mean)/self.image_std)
        H=((max(h for h,w in sizes)+31)//32)*32
        W=((max(w for h,w in sizes)+31)//32)*32
        batch=torch.stack([F.pad(x,(0,W-x.shape[-1],0,H-x.shape[-2])) for x in normalized])
        return ImageList(batch,sizes)

    def _targets(self, targets: Sequence[dict], images: ImageList) -> list[dict[str,Tensor]]:
        if len(targets)!=len(images.image_sizes):
            raise ValueError('One target dictionary per image is required')
        checked=[]
        for target in targets:
            if 'boxes' not in target or 'labels' not in target:
                raise ValueError('Each target needs boxes [N,5] and labels [N]')
            boxes=target['boxes'].to(device=images.tensors.device,dtype=torch.float32)
            labels=target['labels']
            validate_rboxes(boxes)
            if labels.ndim!=1 or len(labels)!=len(boxes) or labels.is_floating_point():
                raise ValueError('labels must be a length-N integral tensor')
            labels=labels.to(device=boxes.device,dtype=torch.long)
            if (labels<0).any() or (labels>=self.config.num_classes).any():
                raise ValueError('Public foreground labels must lie in [0,num_classes-1]')
            checked.append({'boxes':canonicalize(boxes),'labels':labels})
        return checked

    def _compute(self, images: Tensor | Sequence[Tensor], targets: Sequence[dict] | None):
        cfg=self.config
        image_list=self._prepare_images(images)
        gt=None if targets is None else self._targets(targets,image_list)
        features=self.backbone(image_list.tensors)


        with torch.autocast(device_type=image_list.tensors.device.type,enabled=False):
            float_features=OrderedDict((k,v.float()) for k,v in features.items())
            rpn_gt=None if gt is None else [{'boxes':rbox_to_hbb(t['boxes'])} for t in gt]
            proposals,rpn_losses=self.rpn(image_list,float_features,rpn_gt)
        thin=[self.thin(features[str(i)]) for i in range(4)]
        shapes={'fpn':{k:list(v.shape) for k,v in features.items()},
                'thin':{str(i):list(v.shape) for i,v in enumerate(thin)},
                'padded_images':list(image_list.tensors.shape)}
        stage_losses={}
        outputs=[]
        for index,horizontal in enumerate(proposals):
            target=None if gt is None else gt[index]
            if target is not None:
                gt_hbb=rbox_to_hbb(target['boxes'])
                horizontal=torch.cat((horizontal,gt_hbb),0)
            horizontal=horizontal.detach()
            h_refs=hbb_to_rbox(horizontal)
            pooled_h=self.align(thin,h_refs,index)
            delta_h=self.roi_learner(pooled_h.flatten(1))
            learned=decode_boxes(h_refs,delta_h)
            if target is not None:
                labels1,matches1=match_labels(horizontal,gt_hbb,target['labels'],cfg.roi_positive_iou,False)
                positive1,_=balanced_indices(labels1,cfg.roi_batch_size,cfg.roi_positive_fraction)
                if len(positive1):
                    truth1=target['boxes'][matches1[positive1]]
                    code1=equivalent_code_loss(delta_h[positive1],h_refs[positive1],truth1).mean()
                    iou1=(1-aligned_rotated_iou(learned[positive1],truth1)).mean() if cfg.use_iou_loss else None
                else:
                    code1=delta_h.sum()*0
                    iou1=learned.sum()*0 if cfg.use_iou_loss else None
                stage_losses.setdefault('loss_learner_code',[]).append(code1)
                if cfg.use_iou_loss:
                    stage_losses.setdefault('loss_learner_iou',[]).append(cfg.iou_weight*iou1)


            references=learned.detach()
            if target is not None:
                references=torch.cat((references,target['boxes']),0)
            pooled=self.align(thin,references,index,(cfg.context_long,cfg.context_short))
            tokens=self.roi_head(pooled.flatten(1))
            if cfg.use_relation:
                tokens=self.relation(tokens,references)
            logits=self.classifier(tokens)
            deltas=self.regressor(tokens)
            boxes=decode_boxes(references,deltas)
            if target is not None:
                labels,matches=match_labels(references,target['boxes'],target['labels'],cfg.roi_positive_iou,True)
                positive,sampled=balanced_indices(labels,cfg.roi_batch_size,cfg.roi_positive_fraction)
                cls=F.cross_entropy(logits[sampled].float(),labels[sampled]) if len(sampled) else logits.sum()*0
                if len(positive):
                    truth=target['boxes'][matches[positive]]
                    code=equivalent_code_loss(deltas[positive],references[positive],truth).mean()
                    iou=(1-aligned_rotated_iou(boxes[positive],truth)).mean() if cfg.use_iou_loss else None
                else:
                    code=deltas.sum()*0
                    iou=boxes.sum()*0 if cfg.use_iou_loss else None
                stage_losses.setdefault('loss_classifier',[]).append(cls)
                stage_losses.setdefault('loss_fine_code',[]).append(code)
                if cfg.use_iou_loss:
                    stage_losses.setdefault('loss_fine_iou',[]).append(cfg.iou_weight*iou)
            outputs.append({'boxes':boxes,'logits':logits,'deltas':deltas,'references':references,
                'learner_boxes':learned,'horizontal_proposals':horizontal,'roi_features':tokens,
                'aligned_features':pooled,'image_size':image_list.image_sizes[index]})
        losses=dict(rpn_losses)
        losses.update({k:torch.stack(values).mean() for k,values in stage_losses.items()})
        return losses,outputs,shapes

    def forward_train(self, images: Tensor | Sequence[Tensor], targets: Sequence[dict],
                      return_outputs: bool=False):
        self.train()
        losses,outputs,shapes=self._compute(images,targets)
        return (losses,outputs,shapes) if return_outputs else losses

    @torch.no_grad()
    def raw_predictions(self, images: Tensor | Sequence[Tensor]):
        previous=self.training
        self.eval()
        try:
            _,outputs,shapes=self._compute(images,None)
            return outputs,shapes
        finally:
            self.train(previous)

    @torch.no_grad()
    def predict(self, images: Tensor | Sequence[Tensor]) -> list[dict[str,Tensor]]:
        outputs,_=self.raw_predictions(images)
        cfg=self.config
        return [postprocess(x['boxes'],x['logits'].float().softmax(-1),cfg.score_threshold,
                cfg.final_nms_threshold,cfg.max_detections,cfg.pre_nms_topk) for x in outputs]

    def forward(self, images: Tensor | Sequence[Tensor], targets: Sequence[dict] | None=None):
        if self.training:
            if targets is None:
                raise ValueError('Training requires targets; use predict(images) for inference')
            return self.forward_train(images,targets)
        return self.predict(images)


def build_model(config: ModelConfig | dict | str) -> GeometryRelationRoITransformer:
    return GeometryRelationRoITransformer(ModelConfig.from_value(config))
