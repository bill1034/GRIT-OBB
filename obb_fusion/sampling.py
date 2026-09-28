from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from .geometry import _precision


def sample_points(boxes: Tensor, output_size: int = 7, sampling_ratio: int = 2,
                  context: Sequence[float] = (1.,1.)) -> Tensor:

    if output_size<1 or sampling_ratio<1 or len(context)!=2 or min(context)<=0:
        raise ValueError("Positive grid sizes and two positive context scales are required")
    b=_precision(boxes)
    k,s=output_size,sampling_ratio
    iy,ix=torch.meshgrid(torch.arange(k,device=b.device,dtype=b.dtype),
                        torch.arange(k,device=b.device,dtype=b.dtype),indexing='ij')
    sy,sx=torch.meshgrid(torch.arange(s,device=b.device,dtype=b.dtype),
                        torch.arange(s,device=b.device,dtype=b.dtype),indexing='ij')
    ux=(ix.flatten()[:,None]+(sx.flatten()[None,:]+.5)/s)/k-.5
    uy=(iy.flatten()[:,None]+(sy.flatten()[None,:]+.5)/s)/k-.5
    local_x=ux[None]*b[:,None,None,2]*context[0]
    local_y=uy[None]*b[:,None,None,3]*context[1]
    co,si=b[:,None,None,4].cos(),b[:,None,None,4].sin()
    x=b[:,None,None,0]+co*local_x-si*local_y
    y=b[:,None,None,1]+si*local_x+co*local_y
    return torch.stack((x,y),-1)


def rps_pool_single(feature: Tensor, boxes: Tensor, stride: int,
                    output_size: int=7, channels: int=10,
                    sampling_ratio: int=2, context: Sequence[float]=(1.,1.)) -> Tensor:


    k=output_size
    if feature.ndim!=3 or feature.shape[0]!=k*k*channels:
        raise ValueError("Thin feature must have K*K*channels channels")
    if stride<=0:
        raise ValueError("stride must be positive")
    if len(boxes)==0:
        return _precision(feature).new_empty((0,channels,k,k))
    with torch.autocast(device_type=feature.device.type,enabled=False):
        f,b=_precision(feature),_precision(boxes)
        points=sample_points(b,k,sampling_ratio,context)/stride-.5
        x,y=points[...,0],points[...,1]
        x0,y0=x.floor(),y.floor()
        wx,wy=x-x0,y-y0
        H,W=f.shape[-2:]
        channel=torch.arange(k*k*channels,device=f.device).reshape(1,k*k,1,channels)
        flat=f.reshape(-1)
        total=f.new_zeros((len(b),k*k,sampling_ratio**2,channels))
        for dx,dy,weight in [(0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),
                             (0,1,(1-wx)*wy),(1,1,wx*wy)]:
            xx=(x0+dx).long(); yy=(y0+dy).long()
            valid=(xx>=0)&(xx<W)&(yy>=0)&(yy<H)
            index=(channel*H+yy.clamp(0,H-1)[...,None])*W+xx.clamp(0,W-1)[...,None]
            value=flat[index]
            total=total+value*(weight*valid)[...,None]
        return total.mean(2).permute(0,2,1).reshape(len(b),channels,k,k)


class MultiScaleRPSAlign(nn.Module):
    def __init__(self, output_size: int=7, channels: int=10, sampling_ratio: int=2,
                 strides: Sequence[int]=(4,8,16,32)) -> None:
        super().__init__()
        self.output_size=output_size
        self.channels=channels
        self.sampling_ratio=sampling_ratio
        self.strides=tuple(strides)
        if self.strides!=(4,8,16,32):
            raise ValueError("This FPN mapper is defined for P2-P5 strides (4,8,16,32)")

    def forward(self, features: Sequence[Tensor], boxes: Tensor, batch_index: int,
                context: Sequence[float]=(1.,1.)) -> Tensor:
        if len(features)!=len(self.strides):
            raise ValueError("Exactly four P2-P5 feature maps are required")
        output=_precision(features[0]).new_zeros((len(boxes),self.channels,self.output_size,self.output_size))
        if not len(boxes):
            return output


        with torch.no_grad():
            size=(boxes[:,2]*boxes[:,3]).clamp_min(1e-12).sqrt()
            levels=(4+torch.log2(size/224+1e-12)).floor().clamp(2,5).long()-2
        for level,stride in enumerate(self.strides):
            idx=torch.where(levels==level)[0]
            if len(idx):
                pooled=rps_pool_single(features[level][batch_index],boxes[idx],stride,
                    self.output_size,self.channels,self.sampling_ratio,context)
                output=output.index_copy(0,idx,pooled)
        return output
