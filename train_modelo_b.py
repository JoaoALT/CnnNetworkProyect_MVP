"""
Reentrena el Modelo B (ResNet-50 con transfer learning) con la misma receta del notebook
Proyecto_Flores_Oxford102_v2.ipynb (secciones 3, 5, 7 y 8) y deja listos los archivos de la app:

    app/model/model_b_best.pt   pesos entrenados
    app/model/config.json       temperatura de calibración, umbral de confianza y métricas de prueba

Después hay que correr export_onnx.py para generar app/model/flower_model_b.onnx.

Uso (desde la raíz del repo, con GPU):
    python train_modelo_b.py

Diferencia con el notebook: en GPUs de 4 GB el batch de 64 no cabe, así que se procesan
micro-batches de 32 y se acumulan 2 pasos (batch efectivo = 64, mismo número de pasos del optimizador).
"""

import copy
import hashlib
import json
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import loadmat
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import Flowers102

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "app" / "model"

SEED = 42
IMG_SIZE, RESIZE_SHORT_SIDE = 224, 256
BATCH_SIZE, MICRO_BATCH = 64, 32
ACCUM = BATCH_SIZE // MICRO_BATCH
NUM_WORKERS = 4
VAL_FRAC = TEST_FRAC = 0.15
EPOCHS_HEAD, EPOCHS_FT, PATIENCE = 5, 12, 8
HEAD_LR, HEAD_DROPOUT, HEAD_WD = 1e-2, 0.0, 1e-1   # mejores hiperparámetros de la sección 7
CONF_THRESHOLD = 0.5
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
    "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"]
NUM_CLASSES = len(CLASS_NAMES)


# ----------------------------------------------------------------------------- datos

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_data():
    """Descarga, quita duplicados, redimensiona y parte 70/15/15 igual que el notebook (secciones 3 y 5.1)."""
    Flowers102(DATA_ROOT, split="train", download=True)
    flowers_dir = DATA_ROOT / "flowers-102"
    img_dir = flowers_dir / "jpg"
    labels = loadmat(flowers_dir / "imagelabels.mat")["labels"][0] - 1
    df = pd.DataFrame({"filename": [f"image_{i:05d}.jpg" for i in range(1, len(labels) + 1)], "label": labels})
    df["md5"] = [hashlib.md5((img_dir / f).read_bytes()).hexdigest() for f in df["filename"]]
    df = df.drop_duplicates("md5").reset_index(drop=True)

    resized_dir = DATA_ROOT / f"flowers-102-{RESIZE_SHORT_SIDE}"
    resized_dir.mkdir(parents=True, exist_ok=True)
    for f in df["filename"]:
        out = resized_dir / f
        if out.exists():
            continue
        with Image.open(img_dir / f) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = RESIZE_SHORT_SIDE / min(w, h)
            im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC).save(out, quality=95)

    train_df, temp_df = train_test_split(df, test_size=VAL_FRAC + TEST_FRAC, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
                                       stratify=temp_df["label"], random_state=SEED)
    return [d.reset_index(drop=True) for d in (train_df, val_df, test_df)], resized_dir


class AddGaussianNoise:
    """Ruido gaussiano sobre el tensor (valores en [0, 1]), aplicado con probabilidad p."""
    def __init__(self, std=0.08, p=0.2):
        self.std, self.p = std, p

    def __call__(self, t):
        if random.random() < self.p:
            t = (t + torch.randn_like(t) * self.std).clamp(0, 1)
        return t


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.5, 3.0))], p=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0.08, 0.2),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize(RESIZE_SHORT_SIDE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FlowerDataset(Dataset):
    def __init__(self, frame, transform, root):
        self.files, self.labels = frame["filename"].tolist(), frame["label"].tolist()
        self.transform, self.root = transform, Path(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as im:
            return self.transform(im.convert("RGB")), self.labels[i]


# ----------------------------------------------------------------------------- entrenamiento

def build_model(dropout=HEAD_DROPOUT):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_full = len(loader) // ACCUM * ACCUM          # el micro-batch sobrante del final se descarta
    for i, (x, y) in enumerate(loader):
        if i >= n_full:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss / ACCUM).backward()
        if (i + 1) % ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def get_logits(model, loader, device, tta=False):
    """Logits de cada imagen. Con tta=True también devuelve los de la imagen reflejada."""
    model.eval()
    L, LF, Y = [], [], []
    for x, y in loader:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            L.append(model(x).float().cpu())
            if tta:
                LF.append(model(torch.flip(x, dims=[3])).float().cpu())
        Y.append(y)
    return torch.cat(L).numpy(), (torch.cat(LF).numpy() if tta else None), torch.cat(Y).numpy()


def fit(model, name, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    scaler = torch.amp.GradScaler("cuda")
    best_acc, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        lr = optimizer.param_groups[-1]["lr"]
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device)
        lv, _, yv = get_logits(model, val_loader, device)
        va = float((lv.argmax(1) == yv).mean())
        vl = log_loss(yv, softmax(lv, 1), labels=list(range(NUM_CLASSES)))
        flag = ""
        if va > best_acc:
            best_acc, bad, flag = va, 0, "  *mejor*"
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"[{name}] época {ep:02d}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s{flag}", flush=True)
        if bad >= PATIENCE:
            print("Early stopping.", flush=True)
            break
    model.load_state_dict(best_state)
    return best_acc, time.time() - t0


# ----------------------------------------------------------------------------- calibración y métricas

def calibrated_probs(logits, logits_flip, T):
    """Lo mismo que hace la app: softmax(logits / T), promediado con la imagen reflejada."""
    p = softmax(logits / T, axis=1)
    if logits_flip is not None:
        p = (p + softmax(logits_flip / T, axis=1)) / 2
    return p


def fit_temperature(logits, logits_flip, y, grid=np.linspace(0.05, 3.0, 296)):
    losses = [log_loss(y, calibrated_probs(logits, logits_flip, T), labels=list(range(NUM_CLASSES))) for T in grid]
    return float(grid[int(np.argmin(losses))])


def ece(probs, y, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    ids = np.clip(np.digitize(conf, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    return float(sum((ids == b).mean() * abs((pred[ids == b] == y[ids == b]).mean() - conf[ids == b].mean())
                     for b in range(n_bins) if (ids == b).any()))


def metrics(probs, y):
    pred = probs.argmax(1)
    keep = probs.max(1) >= CONF_THRESHOLD
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top5_accuracy": float(top_k_accuracy_score(y, probs, k=5, labels=list(range(NUM_CLASSES)))),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1), labels=list(range(NUM_CLASSES)))),
        "ece": ece(probs, y),
        "errores": int((pred != y).sum()),
        "n": int(len(y)),
        "cobertura_umbral": float(keep.mean()),
        "accuracy_sobre_umbral": float((pred[keep] == y[keep]).mean()) if keep.any() else None,
    }


# ----------------------------------------------------------------------------- main

def main():
    assert torch.cuda.is_available(), "Este script necesita GPU (CUDA)."
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    seed_everything()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    (train_df, val_df, test_df), resized_dir = prepare_data()
    print(f"Entrenamiento: {len(train_df)} | Validación: {len(val_df)} | Prueba: {len(test_df)}", flush=True)

    class_counts = train_df["label"].value_counts().sort_index().values
    weights = torch.as_tensor(1.0 / class_counts[train_df["label"].values], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(FlowerDataset(train_df, train_tf, resized_dir), batch_size=MICRO_BATCH,
                              sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(FlowerDataset(val_df, eval_tf, resized_dir), batch_size=64, **kw)
    test_loader = DataLoader(FlowerDataset(test_df, eval_tf, resized_dir), batch_size=64, **kw)
    steps_per_epoch = len(train_loader) // ACCUM

    model = build_model().to(device).to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Fase 1: solo la cabeza (backbone congelado)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("fc.")
    opt1 = torch.optim.AdamW(model.fc.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_HEAD * steps_per_epoch)
    _, t1 = fit(model, "fase1", train_loader, val_loader, crit, opt1, sch1, EPOCHS_HEAD, device)

    # Fase 2: fine-tuning de toda la red con learning rates diferenciados
    for p in model.parameters():
        p.requires_grad = True
    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt2 = torch.optim.AdamW([{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=[1e-4, 1e-3], epochs=EPOCHS_FT,
                                               steps_per_epoch=steps_per_epoch, pct_start=0.2)
    best_val, t2 = fit(model, "fase2", train_loader, val_loader, crit, opt2, sch2, EPOCHS_FT, device)

    # Calibración (en validación) con el mismo procedimiento que usa la app: temperatura + TTA espejo
    lv, lvf, yv = get_logits(model, val_loader, device, tta=True)
    lt, ltf, yt = get_logits(model, test_loader, device, tta=True)
    T = fit_temperature(lv, lvf, yv)
    report = {
        "sin calibrar, sin TTA": metrics(softmax(lt, 1), yt),
        "sin calibrar, con TTA": metrics(calibrated_probs(lt, ltf, 1.0), yt),
        f"calibrado (T={T:.2f}), con TTA": metrics(calibrated_probs(lt, ltf, T), yt),
    }
    print(f"\nMejor val acc: {best_val:.3f} | temperatura: {T:.2f} | "
          f"tiempo: {(t1 + t2) / 60:.1f} min", flush=True)
    print(pd.DataFrame(report).round(4).to_string(), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(state, OUT_DIR / "model_b_best.pt")
    torch.save(state, CKPT_DIR / "modelB_resnet50_local.pt")
    final = report[f"calibrado (T={T:.2f}), con TTA"]
    config = {
        "model": "resnet50", "class_names": CLASS_NAMES, "img_size": IMG_SIZE, "resize": RESIZE_SHORT_SIDE,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "temperature": T, "tta_flip": True, "conf_threshold": CONF_THRESHOLD,
        "test_metrics": final, "best_val_acc": best_val,
        "trained": {"date": date.today().isoformat(), "gpu": torch.cuda.get_device_name(0),
                    "script": "train_modelo_b.py", "minutes": round((t1 + t2) / 60, 1)},
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {OUT_DIR / 'model_b_best.pt'} y {OUT_DIR / 'config.json'}", flush=True)


if __name__ == "__main__":
    main()
