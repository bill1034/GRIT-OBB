from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .geometry import covariance, gaussian_w2, _precision


def gaussian_position_encoding(boxes: Tensor, dimension: int) -> Tensor:

    if dimension%4:
        raise ValueError("Gaussian PE dimension must be divisible by four")
    b=_precision(boxes)
    k=dimension//4
    frequencies=10000**(-torch.arange(k,device=b.device,dtype=b.dtype)/k)
    mu=b[:,:2,None]*frequencies
    variance=covariance(b).diagonal(dim1=-2,dim2=-1)[:,:,None]*frequencies.square()
    attenuation=torch.exp(-.5*variance)
    return torch.cat((mu.sin()*attenuation,mu.cos()*attenuation),-1).flatten(1)


@torch.no_grad()
def geometric_attention_bias(boxes: Tensor) -> Tensor:


    b=_precision(boxes.detach())
    distance=gaussian_w2(b,b)
    distance=distance.masked_fill(torch.eye(len(b),device=b.device,dtype=torch.bool),0)
    area=b[:,2]*b[:,3]
    scale=(area[:,None]*area[None]).clamp_min(1e-24).pow(.25)
    return -torch.log1p(distance/scale)


class GaussianRelationResidual(nn.Module):

    def __init__(self, input_dim: int=1024, relation_dim: int=128, heads: int=4,
                 gate_init: float=.1) -> None:
        super().__init__()
        if relation_dim%heads or relation_dim%4 or not (-1<gate_init<1):
            raise ValueError("Invalid relation dimensions, head count or gate initialization")
        self.relation_dim=relation_dim
        self.heads=heads
        self.norm=nn.LayerNorm(input_dim)
        self.qkv=nn.Linear(input_dim,3*relation_dim)
        self.output=nn.Linear(relation_dim,input_dim)
        self.raw_tau=nn.Parameter(torch.full((heads,),math.log(math.expm1(1.))))
        self.raw_gate=nn.Parameter(torch.tensor(math.atanh(gate_init)))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: Tensor, boxes: Tensor) -> Tensor:
        if len(tokens)==0:
            return tokens
        q,k,v=self.qkv(self.norm(tokens)).chunk(3,-1)
        with torch.no_grad(),torch.autocast(device_type=boxes.device.type,enabled=False):
            b=_precision(boxes.detach())
            pe=gaussian_position_encoding(b,self.relation_dim)
            bias=geometric_attention_bias(b)

        with torch.autocast(device_type=tokens.device.type,enabled=False):
            q,k,v=_precision(q),_precision(k),_precision(v)
            pe=pe.to(q.dtype)
            d=self.relation_dim//self.heads
            q=(q+pe).reshape(-1,self.heads,d).transpose(0,1)
            k=(k+pe).reshape(-1,self.heads,d).transpose(0,1)
            v=v.reshape(-1,self.heads,d).transpose(0,1)
            tau=F.softplus(self.raw_tau).to(q.dtype)
            logits=q@k.transpose(-1,-2)/math.sqrt(d)+tau[:,None,None]*bias
            mixed=(logits.softmax(-1)@v).transpose(0,1).reshape(-1,self.relation_dim)
        residual=self.output(mixed)
        return tokens+torch.tanh(self.raw_gate)*residual
