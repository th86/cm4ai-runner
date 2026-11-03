"""Challenge prediction runner: reference channels + protein FASTA -> 512x512 uint8 protein PNG.

usage: predict.py --inputs-manifest MANIFEST.json --weights BUNDLE --output DIR [--batch-size 64] [--no-tta]

MANIFEST is the JSON array written by the WDL `write_json(inputs)`: one record per cell with
prediction_id, microtubules_image, nucleus_image, er_image, fasta_file and an optional mask_image.
BUNDLE is the tar/zip made by make_weights.py (model.pt, config.json, antibodies.txt, sequences.json).

Each protein is identified by exact amino-acid sequence match against the training antibodies' FASTA
records. Antibodies matching every record of the FASTA are preferred; otherwise any record match counts;
several matches average their embeddings. A protein with no match gets the mean learned embedding.
"""
import argparse, hashlib, json, os, sys, tarfile, tempfile, time, zipfile
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "scripts")]  # container: same dir; repo: ../scripts/model.py
from model import UNet, dihedral, inv_dihedral  # noqa: E402


def seq_key(seq):
    return hashlib.sha256(seq.strip().upper().replace("*", "").encode()).hexdigest()


def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        raise ValueError(f"no sequence in {path}")
    return seqs


def unpack(bundle, dest):
    if os.path.isdir(bundle):
        return bundle
    # check tar first: a tar holding model.pt (itself a zip) also passes zipfile.is_zipfile
    if tarfile.is_tarfile(bundle):
        with tarfile.open(bundle) as t:
            t.extractall(dest, filter="data")
    elif zipfile.is_zipfile(bundle):
        with zipfile.ZipFile(bundle) as z:
            z.extractall(dest)
    else:
        raise ValueError(f"weights bundle is neither tar nor zip: {bundle}")
    return dest


def load_channel(path, res):
    """512x512 PNG (uint8) or float TIFF -> float32 in [0,1] at `res`, matching the training packing:
    area-average down, round to uint8, divide by 255."""
    im = Image.open(path)
    a = np.asarray(im, dtype=np.float32)  # 0-255 for PNG; float TIFFs are already normalized to [0,1]
    if im.mode == "F":
        a = a * 255.0
    if a.shape != (512, 512):
        raise ValueError(f"{path}: expected 512x512, got {a.shape}")
    f = 512 // res  # same order as prep.py: average raw values, round to uint8, then scale
    a = a.reshape(res, f, res, f).mean((1, 3))
    return np.clip(np.round(a), 0, 255) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-manifest", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.output, exist_ok=True)
    records = json.load(open(a.inputs_manifest))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmp:
        b = unpack(a.weights, tmp)
        cfg = json.load(open(os.path.join(b, "config.json")))
        antibodies = open(os.path.join(b, "antibodies.txt")).read().split()
        seq_index = json.load(open(os.path.join(b, "sequences.json")))  # {antibody: [sha256 of each FASTA record]}
        sd = torch.load(os.path.join(b, "model.pt"), map_location="cpu")
    model = UNet(len(antibodies), w=cfg["width"], emb=cfg["emb"])
    model.load_state_dict(sd, strict=False)
    use_dom = "dom.weight" in sd  # older checkpoints were trained without the domain token
    model.to(dev).eval()
    res = cfg["res"]

    # protein lookup: sequence set -> embedding (cached per FASTA content)
    by_key = {}
    for i, ab in enumerate(antibodies):
        for k in seq_index.get(ab, []):
            by_key.setdefault(k, set()).add(i)
    emb_w = model.emb.weight.detach()
    mean_emb = emb_w.mean(0)
    cache, stats = {}, {"exact": 0, "partial": 0, "unknown": 0}

    def protein_embedding(fasta):
        keys = tuple(sorted(seq_key(s) for s in read_fasta(fasta)))
        if keys not in cache:
            hits = [by_key.get(k, set()) for k in keys]
            full = set.intersection(*hits) if hits else set()
            anyhit = set.union(*hits) if hits else set()
            sel = full or anyhit
            stats["exact" if full else "partial" if anyhit else "unknown"] += 1
            cache[keys] = emb_w[sorted(sel)].mean(0) if sel else mean_emb
        return cache[keys]

    ks = [0] if a.no_tta else range(8)
    n = 0
    for s in range(0, len(records), a.batch_size):
        batch = records[s:s + a.batch_size]
        x = []
        for r in batch:
            ch = [load_channel(r[c], res) for c in ("nucleus_image", "microtubules_image", "er_image")]
            m = r.get("mask_image")
            ch.append(load_channel(m, res) if m else np.ones((res, res), np.float32))
            x.append(np.stack(ch))
        x = torch.from_numpy(np.stack(x).astype(np.float32)).to(dev)
        e = torch.stack([protein_embedding(r["fasta_file"]) for r in batch]).to(dev)
        ab = torch.zeros(len(batch), dtype=torch.long, device=dev)  # unused when e is given
        dom = torch.zeros_like(ab) if use_dom else None  # always predict in the KOLF domain
        with torch.no_grad():
            p = sum(inv_dihedral(model(dihedral(x, k), ab, dom, e), k) for k in ks) / len(ks)
            p = F.interpolate(p[:, None], size=512, mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)
        out = np.rint(p.cpu().numpy() * 255).astype(np.uint8)
        for r, img in zip(batch, out):
            Image.fromarray(img, mode="L").save(os.path.join(a.output, f"{r['prediction_id']}.png"))
        n += len(batch)
    print(f"wrote {n} predictions in {time.time() - t0:.1f}s on {dev}; protein lookup over "
          f"{len(cache)} distinct FASTA: {stats}", flush=True)


if __name__ == "__main__":
    main()
