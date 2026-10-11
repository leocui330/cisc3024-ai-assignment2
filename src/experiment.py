"""Reproducible train/freeze/test pipeline. Training never reads test examples.

Run from repository root: python src/experiment.py prepare|train|evaluate
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key, folder in [("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch"),
                    ("TEMP", "tmp"), ("TMP", "tmp")]:
    path = ROOT / "temp" / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
import numpy as np
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from scipy.optimize import minimize_scalar
from metrics import metrics
from network import SmallCNN, normalized_images

RUN = ROOT / "output" / "runs"
DATA = ROOT / "data" / "input"
SEEDS = [11, 22, 33]
PRIOR_STDS = [0.1, 1.0, 10.0]
SPLIT_SEED = 20261011
MC_SAMPLES = 128
VAL_MC_SEED = 701
TEST_MC_SEED = 1701


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RUN.mkdir(parents=True, exist_ok=True)


def dataset(train, download=False):
    FashionMNIST.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    return FashionMNIST(str(DATA), train=train, download=download)


def prepare():
    # torchvision downloads all four IDX resources but this command loads only
    # the official training images/labels. Test examples are not inspected.
    ds = dataset(True, download=True)
    y = ds.targets.numpy()
    generator = np.random.default_rng(SPLIT_SEED)
    tr, va = [], []
    for cls in range(10):
        idx = generator.permutation(np.flatnonzero(y == cls))
        va.extend(idx[:600].tolist())
        tr.extend(idx[600:].tolist())
    tr, va = np.sort(tr), np.sort(va)
    assert len(tr) == 54000 and len(va) == 6000
    assert len(np.intersect1d(tr, va)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(60000))
    split = ROOT / "data" / "processed" / "split.npz"
    split.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split, train=tr, validation=va)
    checksums = {p.name: sha256(p) for p in sorted((DATA / "FashionMNIST" / "raw").glob("*.gz"))}
    write_json(RUN / "data_manifest.json", dict(dataset="Fashion-MNIST", split_seed=SPLIT_SEED,
        train_count=len(tr), validation_count=len(va), split_sha256=sha256(split),
        train_per_class=np.bincount(y[tr]).tolist(), validation_per_class=np.bincount(y[va]).tolist(),
        compressed_file_sha256=checksums, normalized="(pixel/255 - 0.5)/0.5"))
    print("Prepared 54,000/6,000 disjoint stratified train/validation examples. Test not loaded.", flush=True)


@torch.inference_mode()
def image_probabilities(model, x, batch=512):
    model.eval()
    return torch.cat([model(part).softmax(-1) for part in x.split(batch)]).numpy()


@torch.inference_mode()
def features(model, x, batch=512):
    model.eval()
    return torch.cat([model.features(part) for part in x.split(batch)]).detach()


def load_head(kind, path):
    from heads import MAPHead, VBLLHead
    saved = torch.load(path, map_location="cpu", weights_only=True)
    head = (MAPHead if kind == "map" else VBLLHead)(64, 10, saved["prior_std"])
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head


@torch.inference_mode()
def head_probabilities(head, x, samples, seed, batch=1000):
    # Stable fixed ordering. Each chunk uses a distinct local stream.
    return torch.cat([head.predict_proba(part, samples=samples, seed=seed+i)
                      for i, part in enumerate(x.split(batch))]).numpy()


def train_seed(seed, xtr, ytr, xva, yva, cnn_epochs, head_epochs, fingerprint):
    from heads import MAPHead, VBLLHead
    folder = RUN / f"seed_{seed}"
    if (folder / "selection.json").exists():
        if read_json(folder / "run_fingerprint.json") != fingerprint:
            raise RuntimeError(f"Completed seed {seed} has incompatible provenance")
        print(f"Seed {seed} already completed; preserving existing artifacts.", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "run_fingerprint.json", fingerprint)
    start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cnn_epochs)
    history, best = [], float("inf")
    for epoch in range(1, cnn_epochs+1):
        step = time.perf_counter()
        model.train()
        order = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(seed*1000+epoch))
        total = 0.0
        for idx in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(idx)
        r = metrics(image_probabilities(model, xva), yva.numpy())
        history.append(dict(epoch=epoch, training_ce=total/len(xtr), validation_nll=r["nll"],
                            validation_accuracy=r["accuracy"], seconds=time.perf_counter()-step))
        if r["nll"] < best:
            best, selected_epoch = r["nll"], epoch
            torch.save(model.state_dict(), folder / "cnn.pt")
        scheduler.step()
        write_json(folder / "cnn_history.json", history)
        print(f"seed={seed} CNN {epoch}/{cnn_epochs}: val acc={r['accuracy']:.2f}% NLL={r['nll']:.4f} epoch={history[-1]['seconds']:.1f}s", flush=True)
    cnn_seconds = time.perf_counter()-start
    model.load_state_dict(torch.load(folder / "cnn.pt", weights_only=True))
    model.requires_grad_(False)
    htr, hva = features(model, xtr), features(model, xva)
    torch.save(dict(train=htr, validation=hva, train_labels=ytr, validation_labels=yva), folder / "features.pt")
    initial_mu = torch.cat([model.classifier.weight, model.classifier.bias[:, None]], dim=1)
    selected, all_history = {}, []
    head_start = time.perf_counter()
    for kind, Head in [("map", MAPHead), ("vbll", VBLLHead)]:
        best = float("inf")
        for prior_std in PRIOR_STDS:
            head = Head(64, 10, prior_std)
            with torch.no_grad():
                head.mu.copy_(initial_mu)
            opt = torch.optim.Adam(head.parameters(), lr=0.003)
            for epoch in range(1, head_epochs+1):
                head.train()
                order = torch.randperm(len(htr), generator=torch.Generator().manual_seed(seed*10000+epoch))
                total = 0.0
                for idx in order.split(512):
                    opt.zero_grad(set_to_none=True)
                    loss = head.loss(htr[idx], ytr[idx], len(htr))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {kind} loss")
                    loss.backward()
                    opt.step()
                    total += loss.item()*len(idx)
                head.eval()
                probs = head_probabilities(head, hva, MC_SAMPLES, VAL_MC_SEED)
                r = metrics(probs, yva.numpy())
                all_history.append(dict(method=kind, prior_std=prior_std, epoch=epoch,
                    objective=total/len(htr), validation_nll=r["nll"], validation_accuracy=r["accuracy"]))
                if r["nll"] < best:
                    best = r["nll"]
                    selected[kind] = dict(prior_std=prior_std, epoch=epoch, validation_nll=best,
                                          validation_accuracy=r["accuracy"])
                    torch.save(dict(state_dict=head.state_dict(), prior_std=prior_std), folder / f"{kind}.pt")
            print(f"seed={seed} {kind} prior_std={prior_std} finished; best validation NLL={best:.5f}", flush=True)
            write_json(folder / "head_history.json", all_history)
    head_seconds = time.perf_counter()-head_start
    map_head = load_head("map", folder / "map.pt")
    with torch.inference_mode():
        logits = map_head.moments(hva)[0].double()
    def temp_objective(log_t):
        return nn.functional.cross_entropy(logits / np.exp(log_t), yva).item()
    fit = minimize_scalar(temp_objective, bounds=(np.log(.05), np.log(10)), method="bounded")
    if not fit.success:
        raise RuntimeError("Temperature optimization failed")
    selected["map_temperature"] = dict(temperature=float(np.exp(fit.x)), validation_nll=float(fit.fun))
    vbll = load_head("vbll", folder / "vbll.pt")
    p128 = head_probabilities(vbll, hva, 128, VAL_MC_SEED)
    p512 = head_probabilities(vbll, hva, 512, VAL_MC_SEED)
    mc_check = dict(nll128=metrics(p128, yva.numpy())["nll"], nll512=metrics(p512, yva.numpy())["nll"],
        mean_absolute_probability_difference=float(np.abs(p128-p512).mean()),
        maximum_absolute_probability_difference=float(np.abs(p128-p512).max()))
    # Checkpoint reload must preserve predictions exactly for the same stream.
    reload_p = head_probabilities(load_head("vbll", folder / "vbll.pt"), hva[:1000], 128, VAL_MC_SEED)
    assert np.array_equal(reload_p, p128[:1000])
    np.savez_compressed(folder / "validation_predictions.npz", labels=yva.numpy(), map=head_probabilities(map_head,hva,128,VAL_MC_SEED),
                        vbll=p128, vbll_512=p512)
    write_json(folder / "selection.json", dict(seed=seed, cnn_epoch=selected_epoch,
        cnn_validation_nll=min(row["validation_nll"] for row in history), selected=selected,
        cnn_seconds=cnn_seconds, head_seconds=head_seconds, total_seconds=time.perf_counter()-start,
        validation_mc_check=mc_check, checkpoint_reload_exact=True))


def train(args):
    if (RUN / "freeze_manifest.json").exists():
        raise RuntimeError("Experiment is frozen. Preserve it; do not retrain after test evaluation.")
    if not (ROOT / "data" / "processed" / "split.npz").exists():
        prepare()
    inputs = [ROOT / "src" / f for f in ["experiment.py", "heads.py", "metrics.py", "network.py"]]
    inputs += [ROOT / "docs" / "PROTOCOL.md", ROOT / "data" / "processed" / "split.npz"]
    fingerprint = dict(cnn_epochs=args.cnn_epochs, head_epochs=args.head_epochs, threads=args.threads,
        torch_version=str(torch.__version__), seeds=SEEDS, priors=PRIOR_STDS, mc_samples=MC_SAMPLES,
        inputs={p.relative_to(ROOT).as_posix():sha256(p) for p in inputs})
    existing = RUN / "run_fingerprint.json"
    if existing.exists() and read_json(existing) != fingerprint:
        raise RuntimeError("Run inputs differ from recorded provenance; preserve existing run and inspect")
    write_json(existing, fingerprint)
    ds = dataset(True)
    split = np.load(ROOT / "data" / "processed" / "split.npz")
    x = normalized_images(ds.data)
    y = ds.targets
    xtr, xva = x[split["train"]], x[split["validation"]]
    ytr, yva = y[split["train"]], y[split["validation"]]
    write_json(RUN / "environment.json", dict(python=sys.version, platform=platform.platform(),
        torch=torch.__version__, numpy=np.__version__, cpu_threads=torch.get_num_threads(),
        device="cpu", cuda_available=torch.cuda.is_available(), seeds=SEEDS,
        cnn_epochs=args.cnn_epochs, head_epochs=args.head_epochs, mc_samples=MC_SAMPLES,
        cnn_parameters=sum(p.numel() for p in SmallCNN().parameters())))
    for seed in SEEDS:
        train_seed(seed, xtr, ytr, xva, yva, args.cnn_epochs, args.head_epochs, fingerprint)
    selections = [read_json(RUN / f"seed_{seed}" / "selection.json") for seed in SEEDS]
    # Stability tolerance fixed before test use. Fail rather than silently changing the protocol.
    if any(abs(s["validation_mc_check"]["nll128"]-s["validation_mc_check"]["nll512"]) > .001 for s in selections):
        raise RuntimeError("Validation MC NLL stability exceeds 0.001; inspect before final test.")
    files = [ROOT / "docs" / "PROTOCOL.md", ROOT / "data" / "processed" / "split.npz"]
    files += sorted((ROOT / "src").glob("*.py"))
    for seed in SEEDS:
        files += [RUN / f"seed_{seed}" / f for f in ["cnn.pt", "map.pt", "vbll.pt", "selection.json"]]
    write_json(RUN / "freeze_manifest.json", dict(frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        test_evaluated=False, seeds=SEEDS, file_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in files}))
    print("All configurations frozen. Test set has not been evaluated.", flush=True)


def evaluate():
    manifest = read_json(RUN / "freeze_manifest.json")
    for rel, digest in manifest["file_hashes"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"Frozen input changed: {rel}")
    if (RUN / "test_results.json").exists():
        raise RuntimeError("Final results already exist. Recompute metrics from saved predictions instead.")
    start = time.perf_counter()
    ds = dataset(False)
    x, y = normalized_images(ds.data), ds.targets.numpy()
    results = []
    for seed in SEEDS:
        folder = RUN / f"seed_{seed}"
        selection = read_json(folder / "selection.json")
        model = SmallCNN()
        model.load_state_dict(torch.load(folder / "cnn.pt", weights_only=True))
        h = features(model, x)
        map_head, vbll = load_head("map", folder/"map.pt"), load_head("vbll", folder/"vbll.pt")
        with torch.inference_mode():
            t = selection["selected"]["map_temperature"]["temperature"]
            predictions = dict(map=head_probabilities(map_head,h,MC_SAMPLES,TEST_MC_SEED),
                map_temperature=(map_head.moments(h)[0]/t).softmax(-1).numpy(),
                vbll=head_probabilities(vbll,h,MC_SAMPLES,TEST_MC_SEED),
                vbll_mean=vbll.mean_proba(h).numpy())
        np.savez_compressed(folder / "test_predictions.npz", labels=y, **predictions)
        for method, p in predictions.items():
            r = dict(seed=seed, method=method, **metrics(p,y))
            results.append(r)
            print(f"seed={seed} {method}: acc={r['accuracy']:.2f}% NLL={r['nll']:.5f} Brier={r['brier']:.5f} ECE={r['ece']:.3f}%", flush=True)
    aggregate = {}
    for method in ["map", "map_temperature", "vbll", "vbll_mean"]:
        aggregate[method] = {metric:dict(mean=float(np.mean([r[metric] for r in results if r["method"]==method])),
            sample_std=float(np.std([r[metric] for r in results if r["method"]==method],ddof=1)))
            for metric in ["accuracy","nll","brier","ece"]}
    write_json(RUN / "test_results.json", dict(per_seed=results,aggregate=aggregate,
        evaluation_seconds=time.perf_counter()-start, freeze_manifest_sha256=sha256(RUN / "freeze_manifest.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cnn-epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=40)
    args = parser.parse_args()
    configure(args.threads)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "train":
        train(args)
    else:
        evaluate()
