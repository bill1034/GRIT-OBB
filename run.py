import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from obb_fusion import ModelConfig, build_model, save_checkpoint, load_checkpoint
from obb_fusion.geometry import rbox_to_corners
from runtime.data_adapter import MedicalDataset, _formal_resize_and_pad
from runtime.evaluation import evaluate


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def dataset(args, split):
    return MedicalDataset(args.dataset, split, 1024, args.data_root)


def loader(data, shuffle=False):
    return DataLoader(data, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=collate)


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to avoid overwriting a run.")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    training, validation = dataset(args, "train"), dataset(args, "val")
    if not len(training) or not len(validation):
        raise ValueError("Training and validation splits must not be empty.")
    if training.class_names != validation.class_names:
        raise ValueError("Training and validation class definitions differ.")
    model = build_model(ModelConfig(num_classes=len(training.class_names),
        weights=None if args.no_pretrained else "IMAGENET1K_V1",
        score_threshold=.001, max_detections=300)).to(args.device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50 * len(training))
    best, step = -float("inf"), 0
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run_config.json", {
        "model": model.config.to_dict(), "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "class_names": training.class_names,
        "checkpoint_selection": "validation AP50", "batch_size": 1})
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses_this_epoch = []
        for images, targets in loader(training, shuffle=True):
            images = [image.to(args.device) for image in images]
            targets = [{**t, "boxes": t["boxes"].to(args.device),
                        "labels": t["labels"].to(args.device)} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model.forward_train(images, targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            step += 1
            losses_this_epoch.append(float(loss.detach().cpu()))
        rows = evaluate(model, loader(validation), torch.device(args.device),
                        validation.class_names, compute_loss=False)
        ap50 = rows["val_AP50"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses_this_epoch)),
                        "validation_AP50": ap50})
        print(json.dumps(history[-1]), flush=True)
        extra = {"epoch": epoch + 1, "class_names": training.class_names,
                 "validation_AP50": ap50}
        save_checkpoint(args.output / "last.pt", model, step=step, extra=extra)
        if ap50 > best:
            best = ap50
            save_checkpoint(args.output / "best.pt", model, step=step, extra=extra)
        write_json(args.output / "history.json", history)


def evaluate_split(args):
    data = dataset(args, args.split)
    model, meta = load_checkpoint(args.checkpoint, map_location=args.device)
    if meta["extra"].get("class_names") != data.class_names:
        raise ValueError("Checkpoint and dataset class definitions differ.")
    rows = evaluate(model, loader(data), torch.device(args.device),
                    data.class_names, compute_loss=False)
    rows["AP50"] = rows.pop("val_AP50")
    rows["split"] = args.split
    write_json(args.output, rows)
    print(json.dumps({k: v for k, v in rows.items() if k not in ("ground_truth", "predictions")}))


def predict(args):
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score threshold must be between 0 and 1")
    model, meta = load_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    tensor, _, metadata = _formal_resize_and_pad(image, torch.empty((0, 4, 2)), 1024)
    with torch.no_grad():
        prediction = model.predict([tensor.to(args.device)])[0]
    polygons = MedicalDataset.to_original_polygons(rbox_to_corners(prediction["boxes"]).cpu(), metadata)
    names = meta["extra"].get("class_names", [])
    detections = []
    for polygon, label, score in zip(polygons.tolist(), prediction["labels"].tolist(), prediction["scores"].tolist()):
        if score >= args.score_threshold:
            detections.append({"class_id": label, "class_name": names[label] if names else str(label),
                               "score": score, "polygon": polygon})
    write_json(args.output, {"image_size": [image.shape[0], image.shape[1]], "detections": detections})


def main():
    parser = argparse.ArgumentParser(description="GRIT-OBB")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "predict"):
        p = sub.add_parser(command)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--output", type=Path, required=True)
        if command != "predict":
            p.add_argument("--dataset", choices=("focus", "ocd", "au_opg"), required=True)
            p.add_argument("--data-root", type=Path, required=True)
        if command != "train":
            p.add_argument("--checkpoint", type=Path, required=True)
        if command == "train":
            p.add_argument("--epochs", type=int, default=50)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--no-pretrained", action="store_true")
            p.set_defaults(func=train)
        elif command == "evaluate":
            p.add_argument("--split", choices=("val", "test"), default="val")
            p.set_defaults(func=evaluate_split)
        else:
            p.add_argument("--image", type=Path, required=True)
            p.add_argument("--score-threshold", type=float, default=.5)
            p.set_defaults(func=predict)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.func(args)


if __name__ == "__main__":
    main()
