from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import Tensor
import torch.nn.functional as F


def _precision(x: Tensor) -> Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def wrap_angle(theta: Tensor, period: float = math.pi) -> Tensor:

    return torch.remainder(theta + period / 2, period) - period / 2


def canonicalize(boxes: Tensor) -> Tensor:

    if boxes.shape[-1] != 5:
        raise ValueError("rboxes must end in five coordinates")
    b = _precision(boxes)
    swap = b[..., 3] > b[..., 2]
    w = torch.where(swap, b[..., 3], b[..., 2])
    h = torch.where(swap, b[..., 2], b[..., 3])
    a = wrap_angle(b[..., 4] + swap.to(b.dtype) * (math.pi / 2))
    return torch.cat((b[..., :2], w[..., None], h[..., None], a[..., None]), -1)


def validate_rboxes(boxes: Tensor, name: str = "boxes") -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not boxes.is_floating_point():
        raise ValueError(f"{name} must be floating point [N,5]")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} contains nonfinite values")
    if (boxes[:, 2:4] <= 0).any():
        raise ValueError(f"{name} widths and heights must be strictly positive")


def rbox_to_corners(boxes: Tensor) -> Tensor:
    b = _precision(boxes)
    unit = b.new_tensor([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    local = b[..., None, 2:4] * unit
    co, si = b[..., 4].cos()[..., None], b[..., 4].sin()[..., None]
    x = co * local[..., 0] - si * local[..., 1] + b[..., None, 0]
    y = si * local[..., 0] + co * local[..., 1] + b[..., None, 1]
    return torch.stack((x, y), -1)


def rectangle_to_rbox(polygons: Tensor, tolerance: float = 1e-5) -> Tensor:


    if polygons.shape[-2:] != (4, 2):
        raise ValueError("Expected [...,4,2] rectangle vertices")
    p = _precision(polygons)
    e = p.roll(-1, -2) - p
    w, h = e[..., 0, :].norm(dim=-1), e[..., 1, :].norm(dim=-1)
    scale = torch.maximum(w, h).clamp_min(1e-12)
    opposite = (e[..., :2, :] + e[..., 2:, :]).abs().amax(dim=(-2,-1))
    dot = (e[..., 0, :] * e[..., 1, :]).sum(-1).abs()
    if (w <= 0).any() or (h <= 0).any() or (opposite > tolerance * scale).any() or (dot > tolerance * w * h).any():
        raise ValueError("polygon is not a nondegenerate rectangle in cyclic order")
    return canonicalize(torch.cat((p.mean(-2), w[...,None], h[...,None],
        torch.atan2(e[...,0,1], e[...,0,0])[...,None]), -1))


def rbox_to_hbb(boxes: Tensor) -> Tensor:
    p = rbox_to_corners(boxes)
    return torch.cat((p.amin(-2), p.amax(-2)), -1)


def hbb_to_rbox(boxes: Tensor) -> Tensor:

    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(.01)
    return torch.cat((center, wh, torch.zeros_like(wh[..., :1])), -1)


def _cross(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(points: Tensor, box: Tensor, eps: float) -> Tensor:
    d = points - box[:, None, :2]
    co, si = box[:, None, 4].cos(), box[:, None, 4].sin()
    local_x = co*d[...,0] + si*d[...,1]
    local_y = -si*d[...,0] + co*d[...,1]

    return (local_x.abs() <= box[:,None,2]/2 + eps*box[:,None,2]) & (local_y.abs() <= box[:,None,3]/2 + eps*box[:,None,3])


def _aligned_iou_core(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape or a.shape[-1] != 5:
        raise ValueError("aligned IoU requires identically shaped [...,5] tensors")
    shape = a.shape[:-1]
    if a.numel() == 0:
        return a[..., 0] * 0
    aa, bb = a.reshape(-1,5), b.reshape(-1,5)


    origin = aa[:, :2]
    aa = torch.cat((aa[:, :2]-origin, aa[:, 2:]), -1)
    bb = torch.cat((bb[:, :2]-origin, bb[:, 2:]), -1)
    pa, pb = rbox_to_corners(aa), rbox_to_corners(bb)
    ea, eb = pa.roll(-1,1)-pa, pb.roll(-1,1)-pb
    eps = 2e-7 if a.dtype == torch.float32 else 1e-12
    va, vb = _inside(pa,bb,eps), _inside(pb,aa,eps)
    p, r = pa[:,:,None,:], ea[:,:,None,:]
    q, s = pb[:,None,:,:], eb[:,None,:,:]
    denominator = _cross(r,s)
    edge_scale = r.norm(dim=-1)*s.norm(dim=-1)
    parallel = denominator.abs() <= eps*edge_scale.clamp_min(1e-12)
    safe_denominator = torch.where(parallel, torch.ones_like(denominator), denominator)
    t = _cross(q-p,s)/safe_denominator
    u = _cross(q-p,r)/safe_denominator
    good = (~parallel) & (t>=-eps) & (t<=1+eps) & (u>=-eps) & (u<=1+eps)
    intersections = p+t[...,None]*r
    vertices = torch.cat((pa,pb,intersections.flatten(1,2)),1)
    valid = torch.cat((va,vb,good.flatten(1,2)),1)
    vertices = torch.where(valid[...,None],vertices,torch.zeros_like(vertices))
    count = valid.sum(1)
    centroid = vertices.sum(1)/count.clamp_min(1)[:,None]
    centered = vertices-centroid[:,None]


    with torch.no_grad():
        angles = torch.atan2(centered[...,1],centered[...,0])
        angles = torch.where(valid,angles,torch.full_like(angles,4*math.pi))
        order = angles.argsort(dim=1,stable=True)
    v = centered.gather(1,order[...,None].expand(-1,-1,2))
    idx = torch.arange(v.shape[1],device=a.device)[None,:].expand(len(v),-1)
    nxt = torch.where(idx+1<count[:,None],idx+1,torch.zeros_like(idx))
    vnext = v.gather(1,nxt[...,None].expand(-1,-1,2))
    area = (_cross(v,vnext)*(idx<count[:,None])).sum(1).abs()/2
    area = torch.where(count>=3,area,torch.zeros_like(area))
    area_a, area_b = aa[:,2]*aa[:,3], bb[:,2]*bb[:,3]

    area = torch.minimum(area,torch.minimum(area_a,area_b)).clamp_min(0)
    union = (area_a+area_b-area).clamp_min(1e-12)
    return (area/union).clamp(0,1).reshape(shape)


def aligned_rotated_iou(a: Tensor, b: Tensor) -> Tensor:

    with torch.autocast(device_type=a.device.type,enabled=False):
        dtype = torch.promote_types(_precision(a).dtype,_precision(b).dtype)
        return _aligned_iou_core(a.to(dtype),b.to(dtype))


def pairwise_rotated_iou(a: Tensor, b: Tensor, chunk_size: int = 64) -> Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[-1] != 5 or b.shape[-1] != 5:
        raise ValueError("pairwise IoU expects [N,5] and [M,5]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if len(a)==0 or len(b)==0:
        return _precision(a).new_zeros((len(a),len(b)))
    blocks=[]
    for start in range(0,len(a),chunk_size):
        x=a[start:start+chunk_size]
        blocks.append(aligned_rotated_iou(x[:,None,:].expand(-1,len(b),-1),b[None,:,:].expand(len(x),-1,-1)))
    return torch.cat(blocks,0)


def _code_variants(references: Tensor, targets: Tensor) -> Tensor:
    ref, gt = _precision(references), _precision(targets)
    d = gt[...,:2]-ref[...,:2]
    co, si = ref[...,4].cos(),ref[...,4].sin()
    tx = (d[...,0]*co+d[...,1]*si)/ref[...,2].clamp_min(1e-12)
    ty = (-d[...,0]*si+d[...,1]*co)/ref[...,3].clamp_min(1e-12)
    xy=torch.stack((tx,ty),-1)
    wh0=(gt[...,2:4]/ref[...,2:4].clamp_min(1e-12)).clamp_min(1e-12).log()
    wh1=(gt[...,[3,2]]/ref[...,2:4].clamp_min(1e-12)).clamp_min(1e-12).log()
    angle0=wrap_angle(gt[...,4]-ref[...,4])[...,None]
    angle1=wrap_angle(gt[...,4]+math.pi/2-ref[...,4])[...,None]
    return torch.stack((torch.cat((xy,wh0,angle0),-1),torch.cat((xy,wh1,angle1),-1)),-2)


def encode_boxes(references: Tensor, targets: Tensor) -> Tensor:

    variants=_code_variants(references,targets)
    best=variants[...,-1].abs().argmin(-1)
    return variants.gather(-2,best[...,None,None].expand(*best.shape,1,5)).squeeze(-2)


def decode_boxes(references: Tensor, deltas: Tensor) -> Tensor:

    with torch.autocast(device_type=references.device.type,enabled=False):
        ref, delta = _precision(references),_precision(deltas)
        wh=ref[...,2:4].clamp_min(.01)
        local=delta[...,:2]*wh
        co,si=ref[...,4].cos(),ref[...,4].sin()
        x=ref[...,0]+local[...,0]*co-local[...,1]*si
        y=ref[...,1]+local[...,0]*si+local[...,1]*co
        new_wh=(wh*delta[...,2:4].clamp(-math.log(1000),math.log(1000)).exp()).clamp_min(.01)
        angle=wrap_angle(ref[...,4]+delta[...,4])
        return canonicalize(torch.cat((torch.stack((x,y),-1),new_wh,angle[...,None]),-1))


def equivalent_code_loss(predicted_deltas: Tensor, references: Tensor, targets: Tensor, beta: float = 1/9) -> Tensor:


    p=_precision(predicted_deltas)
    targets2=_code_variants(references,targets)
    error=p[...,None,:]-targets2
    error=torch.cat((error[...,:4],wrap_angle(error[...,4])[...,None]),-1)
    return F.smooth_l1_loss(error,torch.zeros_like(error),beta=beta,reduction='none').sum(-1).amin(-1)


def covariance(boxes: Tensor) -> Tensor:
    b=_precision(boxes)
    co,si=b[...,4].cos(),b[...,4].sin()
    w2,h2=b[...,2].square()/4,b[...,3].square()/4
    xx=co.square()*w2+si.square()*h2
    yy=si.square()*w2+co.square()*h2
    xy=co*si*(w2-h2)
    return torch.stack((xx,xy,xy,yy),-1).reshape(*b.shape[:-1],2,2)


def gaussian_w2(a: Tensor, b: Tensor) -> Tensor:


    with torch.autocast(device_type=a.device.type,enabled=False):
        a,b=_precision(a),_precision(b)
        A,B=covariance(a),covariance(b)
        trA=A.diagonal(dim1=-2,dim2=-1).sum(-1)
        trB=B.diagonal(dim1=-2,dim2=-1).sum(-1)
        traceAB=(A[:,None]*B[None]).sum((-1,-2))
        sqrt_detA=(a[:,2]*a[:,3])/4
        sqrt_detB=(b[:,2]*b[:,3])/4
        trace_sqrt=(traceAB+2*sqrt_detA[:,None]*sqrt_detB[None]).clamp_min(0).sqrt()
        cov_distance=(trA[:,None]+trB[None]-2*trace_sqrt).clamp_min(0)
        mean_distance=(a[:,None,:2]-b[None,:,:2]).square().sum(-1)
        return (mean_distance+cov_distance).clamp_min(0).sqrt()


def transform_polygons(polygons: Tensor, scale: float | Sequence[float], pad: Sequence[float]=(0.,0.)) -> Tensor:

    s=(float(scale),float(scale)) if isinstance(scale,(float,int)) else tuple(scale)
    if len(s)!=2 or min(s)<=0 or len(pad)!=2:
        raise ValueError("scale must be positive (sx,sy); pad must be (x,y)")
    return polygons*polygons.new_tensor(s)+polygons.new_tensor(pad)


def transform_rboxes(boxes: Tensor, scale: float | Sequence[float], pad: Sequence[float]=(0.,0.)) -> Tensor:

    s=(float(scale),float(scale)) if isinstance(scale,(float,int)) else tuple(scale)
    if len(s)!=2 or min(s)<=0 or abs(s[0]-s[1])>1e-12:
        raise ValueError("Anisotropic resize does not preserve rotated rectangles; transform polygons instead")
    if len(pad)!=2:
        raise ValueError("pad must be (x,y)")
    b=_precision(boxes)
    return canonicalize(torch.cat((b[...,:2]*s[0]+b.new_tensor(pad),b[...,2:4]*s[0],b[...,4:]),-1))
