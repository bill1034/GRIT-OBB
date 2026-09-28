from __future__ import annotations
import torch
from torch import Tensor
from .geometry import (aligned_rotated_iou, pairwise_rotated_iou, rbox_to_corners,
                       rbox_to_hbb, canonicalize)


@torch.no_grad()
def rotated_nms(boxes: Tensor, scores: Tensor, iou_threshold: float=.5) -> Tensor:
    if not 0<=iou_threshold<=1 or len(boxes)!=len(scores):
        raise ValueError("Invalid IoU threshold or mismatched scores")
    if not len(boxes):
        return torch.empty(0,dtype=torch.long,device=boxes.device)
    order=scores.argsort(descending=True,stable=True)
    envelopes=rbox_to_hbb(boxes)
    keep=[]
    while len(order):
        current=order[0]
        keep.append(current)
        rest=order[1:]
        if not len(rest):
            break


        left=torch.maximum(envelopes[current,:2],envelopes[rest,:2])
        right=torch.minimum(envelopes[current,2:],envelopes[rest,2:])
        candidates=(right>left).all(-1)
        overlaps=scores.new_zeros(len(rest))
        idx=torch.where(candidates)[0]
        if len(idx):
            overlaps[idx]=pairwise_rotated_iou(boxes[current][None],boxes[rest[idx]]).squeeze(0)
        order=rest[overlaps<=iou_threshold]
    return torch.stack(keep)


@torch.no_grad()
def class_aware_rotated_nms(boxes: Tensor, scores: Tensor, labels: Tensor,
                           iou_threshold: float=.5,
                           max_detections: int | None=None) -> Tensor:
    if max_detections is not None and max_detections < 1:
        raise ValueError("max_detections must be positive")
    if max_detections is not None:
        if not len(boxes):
            return torch.empty(0,dtype=torch.long,device=boxes.device)


        order=torch.arange(len(scores),device=scores.device)
        order=order[labels[order].argsort(stable=True)]
        order=order[scores[order].argsort(descending=True,stable=True)]
        envelopes=rbox_to_hbb(boxes)
        keep=[]
        kept_by_class={}
        for current in order:
            previous=kept_by_class.get(int(labels[current]))
            suppressed=False
            if previous:
                prior=torch.stack(previous)
                left=torch.maximum(envelopes[current,:2],envelopes[prior,:2])
                right=torch.minimum(envelopes[current,2:],envelopes[prior,2:])
                candidates=(right>left).all(-1)
                idx=torch.where(candidates)[0]
                if len(idx):
                    overlaps=aligned_rotated_iou(
                        boxes[prior[idx]],boxes[current][None].expand(len(idx),-1))
                    suppressed=bool((overlaps>iou_threshold).any())
            if not suppressed:
                keep.append(current)
                kept_by_class.setdefault(int(labels[current]),[]).append(current)
                if len(keep)==max_detections:
                    break
        return torch.stack(keep)
    keeps=[]
    for label in labels.unique():
        idx=torch.where(labels==label)[0]
        keeps.append(idx[rotated_nms(boxes[idx],scores[idx],iou_threshold)])
    if not keeps:
        return torch.empty(0,dtype=torch.long,device=boxes.device)
    keep=torch.cat(keeps)
    return keep[scores[keep].argsort(descending=True,stable=True)]


@torch.no_grad()
def postprocess(boxes: Tensor, probabilities: Tensor, score_threshold: float=.05,
                iou_threshold: float=.5, max_detections: int=1000,
                pre_nms_topk: int=5000) -> dict[str,Tensor]:
    if not torch.isfinite(boxes).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite detector output; refusing to hide it with filtering")

    proposal,category=torch.where(probabilities[:,1:]>score_threshold)
    scores=probabilities[proposal,category+1]
    if len(scores)>pre_nms_topk:
        order=scores.argsort(descending=True,stable=True)[:pre_nms_topk]
        proposal,category,scores=proposal[order],category[order],scores[order]
    selected=canonicalize(boxes[proposal])
    keep=class_aware_rotated_nms(selected,scores,category,iou_threshold,max_detections)
    selected=selected[keep]
    return {'boxes':selected,'polygons':rbox_to_corners(selected),'scores':scores[keep],
            'labels':category[keep].long(),'proposal_indices':proposal[keep].long()}
