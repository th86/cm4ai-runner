"""Challenge prediction runner: reference channels + protein FASTA -> 512x512 uint8 protein PNG.

usage: predict.py --inputs-manifest MANIFEST.json --weights BUNDLE --output DIR [--batch-size 64] [--no-tta]

MANIFEST is the JSON array written by the WDL `write_json(inputs)`: one record per cell with
prediction_id, microtubules_image, nucleus_image, er_image, fasta_file and an optional mask_image.
BUNDLE is the tar/zip made by make_weights.py (model.pt [, model_1.pt, ...], config.json, antibodies.txt,
sequences.json). Several model files form an ensemble: each uses its own embedding table and predictions
are averaged (after TTA, before brightness shrinkage).

Each protein is identified by exact amino-acid sequence match against the training antibodies' FASTA
records. Antibodies matching every record of the FASTA are preferred; otherwise any record match counts;
several matches average their embeddings. A protein with no match gets the mean learned embedding.

If config.json has a "shrink" block, proteins matched only to antibodies without KOLF training cells have
their brightness pulled toward the KOLF average: each such protein's predictions are multiplied by
f = (1 - lambda) + lambda * b_avg / b_pred, where b_pred is the protein's mean prediction over all its cells
in this run (cells of one protein share a FASTA). This needs two passes; predictions are kept at model
resolution in between.

If the bundle has esm.npz, a protein whose FASTA matches no antibody gets the mean embedding of the K
vocabulary proteins with the most similar ESM-2 sequence vectors (cosine on standardised vectors) instead
of the mean of all embeddings, and no brightness shrinkage. The ESM-2 model is read from $CM4AI_ESM_DIR
(the image bakes it into /opt/esm2) or else downloaded by name.

If the bundle has annot.json (v5), every protein that does not match an antibody with KOLF training cells gets
the mean embedding of its K nearest *other* vocabulary proteins by z-scored ESM-2 cosine plus z-scored Jaccard
overlap of HPA localization annotations (ESM only when unannotated), with no brightness shrinkage. The gene of
a test FASTA is found by exact sequence match to Ensembl; its annotation comes from the HPA atlas table.

If the bundle has a texture library (v6), each prediction gets alpha x a real residual (real minus predicted image
of a KOLF training cell) added: residuals of the protein's own antibody when it has KOLF cells, otherwise of its
K nearest KOLF proteins (same similarity as above). Requests with identical input images are treated as repeated
predictions of one cell and get different residuals (a deterministic, cell-specific shuffle), so several
predictions per cell differ. --no-texture switches this off.
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
    ap.add_argument("--no-texture", action="store_true", help="skip the texture library (smooth predictions)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.output, exist_ok=True)
    records = json.load(open(a.inputs_manifest))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tmpdir = tempfile.TemporaryDirectory()  # kept for the whole run: the texture library is read from it on demand
    b = unpack(a.weights, tmpdir.name)
    cfg = json.load(open(os.path.join(b, "config.json")))
    antibodies = open(os.path.join(b, "antibodies.txt")).read().split()
    seq_index = json.load(open(os.path.join(b, "sequences.json")))  # {antibody: [sha256 of each FASTA record]}
    files = ["model.pt"] + [f"model_{i}.pt" for i in range(1, cfg.get("n_models", 1))]
    sds = [torch.load(os.path.join(b, f), map_location="cpu") for f in files]
    esm_tab = dict(np.load(os.path.join(b, "esm.npz"))) if "esm" in cfg else None
    annot = json.load(open(os.path.join(b, "annot.json"))) if "neighbors" in cfg else None
    tex = None if a.no_texture else cfg.get("texture")
    if tex:
        R = np.load(os.path.join(b, "residuals.npy"), mmap_mode="r")
        row_of = {ab: i for i, ab in enumerate(antibodies)}
        res_of = {}  # vocabulary row -> indices into R
        for j, ab in enumerate(json.load(open(os.path.join(b, "residual_index.json")))["antibody"]):
            res_of.setdefault(row_of[ab], []).append(j)
    models = []
    for sd in sds:
        m = UNet(len(antibodies), w=cfg["width"], emb=cfg["emb"])
        m.load_state_dict(sd, strict=False)
        # older checkpoints were trained without the domain token
        models.append((m.to(dev).eval(), "dom.weight" in sd))
    res = cfg["res"]

    # protein lookup: sequence set -> embedding (cached per FASTA content)
    by_key = {}
    for i, ab in enumerate(antibodies):
        for k in seq_index.get(ab, []):
            by_key.setdefault(k, set()).add(i)
    cache, stats = {}, {"exact": 0, "partial": 0, "esm": 0, "unknown": 0, "neighbors": 0}
    if esm_tab is not None:
        esm_row = {int(r): j for j, r in enumerate(esm_tab["rows"])}
        esm_vec = esm_tab["vec"].astype(np.float32)
    if annot:
        nn_kolf = {i for i, ab in enumerate(antibodies) if ab in set(cfg["neighbors"]["kolf_antibodies"])}
        vocab_locs = [set(v) for v in annot["vocab_locs"]]
    esm_model = []  # loaded on first unknown protein

    def esm_neighbours(fasta):
        if not esm_model:
            from esm_util import ESM
            path = os.environ.get("CM4AI_ESM_DIR") or cfg["esm"]["model"]
            esm_model.append(ESM(path, device=str(dev)))
        v = (esm_model[0].embed_records(read_fasta(fasta)) - esm_tab["mu"]) / esm_tab["sd"]
        sims = esm_tab["vec"].astype(np.float32) @ (v / np.linalg.norm(v)).astype(np.float32)
        return sorted(int(esm_tab["rows"][j]) for j in np.argsort(-sims)[:cfg["esm"]["k"]])
    shrink = cfg.get("shrink")
    kolf_ids = {i for i, ab in enumerate(antibodies) if shrink and ab in set(shrink["kolf_antibodies"])}
    shrink_keys = set()  # proteins matched only to antibodies without KOLF training cells

    def esm_query(fasta, sel):
        """Unit ESM-2 vector of the protein: stored for vocabulary proteins, computed otherwise."""
        rows = [esm_row[r] for r in sel if r in esm_row]
        if rows:
            v = esm_vec[rows].mean(0)
        else:
            if not esm_model:
                from esm_util import ESM
                esm_model.append(ESM(os.environ.get("CM4AI_ESM_DIR") or cfg["esm"]["model"], device=str(dev)))
            v = ((esm_model[0].embed_records(read_fasta(fasta)) - esm_tab["mu"]) / esm_tab["sd"]).astype(np.float32)
        return v / np.linalg.norm(v)

    def nearest(fasta, keys, sel, pool, k):
        """k rows of `pool` (minus the protein's own rows) most similar by z ESM cosine + z annotation Jaccard."""
        cand = [r for r in pool if r not in sel and (esm_tab is None or r in esm_row)]
        if esm_tab is None:
            return cand[:k]
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        score = z(esm_vec[[esm_row[r] for r in cand]] @ esm_query(fasta, sel))
        if annot:
            locs = {l for kk in keys for g in annot["seq2genes"].get(kk, []) for l in annot["gene_locs"].get(g, [])}
            if not locs:
                locs = set().union(*(vocab_locs[r] for r in sel)) if sel else set()
            if locs:
                score = score + z(np.array([len(locs & vocab_locs[r]) / len(locs | vocab_locs[r]) if vocab_locs[r] else 0.0
                                            for r in cand]))
        return sorted(cand[j] for j in np.argsort(-score)[:k])

    def annotation_neighbours(fasta, keys, sel):
        return nearest(fasta, keys, sel, list(esm_row), cfg["neighbors"]["k"])

    donors = {}  # protein -> residual indices supplying its textures

    def texture_donors(fasta, keys, sel):
        own = [j for r in sorted(sel) for j in res_of.get(r, [])]
        if own:
            return own
        rows = nearest(fasta, keys, sel, sorted(res_of), tex["k"])
        return [j for r in rows for j in res_of[r]]

    def protein_embedding(fasta):
        keys = tuple(sorted(seq_key(s) for s in read_fasta(fasta)))
        if keys not in cache:
            hits = [by_key.get(k, set()) for k in keys]
            full = set.intersection(*hits) if hits else set()
            anyhit = set.union(*hits) if hits else set()
            sel = full or anyhit
            if annot is not None and not (sel & nn_kolf):
                stats["neighbors"] += 1
                cache[keys] = annotation_neighbours(fasta, keys, sel)  # no shrinkage for these
            elif sel:
                stats["exact" if full else "partial"] += 1
                cache[keys] = sorted(sel)  # matched embedding rows
            elif esm_tab is not None:
                stats["esm"] += 1
                cache[keys] = esm_neighbours(fasta)  # rows of the most similar sequences; no shrinkage
            else:
                stats["unknown"] += 1
                cache[keys] = []  # -> mean of all rows
            if shrink and sel and not (sel & kolf_ids) and annot is None:  # never shrink neighbour embeddings
                shrink_keys.add(keys)
            if tex:
                donors[keys] = texture_donors(fasta, keys, sel)
        return keys, cache[keys]

    ks = [0] if a.no_tta else range(8)
    P = np.empty((len(records), res, res), np.float16)  # pass 1: predictions at model resolution
    prot = []
    for s in range(0, len(records), a.batch_size):
        batch = records[s:s + a.batch_size]
        x = []
        for r in batch:
            ch = [load_channel(r[c], res) for c in ("nucleus_image", "microtubules_image", "er_image")]
            m = r.get("mask_image")
            ch.append(load_channel(m, res) if m else np.ones((res, res), np.float32))
            x.append(np.stack(ch))
        x = torch.from_numpy(np.stack(x).astype(np.float32)).to(dev)
        looked = [protein_embedding(r["fasta_file"]) for r in batch]
        prot += [k for k, _ in looked]
        ab = torch.zeros(len(batch), dtype=torch.long, device=dev)  # unused when e is given
        p = 0
        with torch.no_grad():
            for model, use_dom in models:
                w = model.emb.weight
                e = torch.stack([w[rows].mean(0) if rows else w.mean(0) for _, rows in looked])
                dom = torch.zeros_like(ab) if use_dom else None  # always predict in the KOLF domain
                p = p + sum(inv_dihedral(model(dihedral(x, k), ab, dom, e), k) for k in ks) / len(ks)
        P[s:s + len(batch)] = (p / len(models)).cpu().numpy().astype(np.float16)

    # brightness factors per protein (1 unless the protein is shrunk)
    factor = np.ones(len(records), np.float32)
    if shrink_keys:
        idx_by = {}
        for i, k in enumerate(prot):
            idx_by.setdefault(k, []).append(i)
        lam, b_avg = shrink["lambda"], shrink["b_avg"]
        for k in shrink_keys:
            ii = idx_by[k]
            b_pred = float(np.clip(P[ii].astype(np.float32), 0, 1).mean())
            factor[ii] = (1 - lam) + lam * b_avg / max(b_pred, 1e-6)

    # repeated requests of one cell (identical input images) get different textures
    sample_of, cell_of, seen = [], [], {}
    for r in records:
        h = hashlib.sha1()
        for c in ("nucleus_image", "microtubules_image", "er_image", "mask_image"):
            if r.get(c):
                h.update(open(r[c], "rb").read())
        ck = h.hexdigest(); cell_of.append(ck); sample_of.append(seen.get(ck, 0)); seen[ck] = sample_of[-1] + 1

    def texture(i):
        d = donors[prot[i]]
        perm = np.random.default_rng(int(hashlib.sha1((cell_of[i] + "".join(prot[i])).encode()).hexdigest()[:12], 16)).permutation(len(d))
        return R[d[perm[sample_of[i] % len(d)]]].astype(np.float32) * (tex["scale"] / 255.0)

    # pass 2: upsample, apply factors and textures, write PNGs
    n = 0
    for s in range(0, len(records), a.batch_size):
        batch = records[s:s + a.batch_size]
        p = torch.from_numpy(P[s:s + len(batch)].astype(np.float32)).to(dev)
        f = torch.from_numpy(factor[s:s + len(batch)]).to(dev)
        with torch.no_grad():
            p = F.interpolate(p[:, None], size=512, mode="bilinear", align_corners=False)[:, 0]
            p = (p * f[:, None, None]).clamp(0, 1)
        p = p.cpu().numpy()
        if tex:
            for q in range(len(batch)):
                p[q] = np.clip(p[q] + tex["alpha"] * texture(s + q), 0, 1)
        out = np.rint(p * 255).astype(np.uint8)
        for r, img in zip(batch, out):
            Image.fromarray(img, mode="L").save(os.path.join(a.output, f"{r['prediction_id']}.png"))
        n += len(batch)
    print(f"wrote {n} predictions in {time.time() - t0:.1f}s on {dev}; protein lookup over "
          f"{len(cache)} distinct FASTA: {stats}; brightness-shrunk proteins: {len(shrink_keys)}; "
          f"models: {len(models)}; textured: {'alpha ' + str(tex['alpha']) if tex else 'no'}; "
          f"cells with repeated requests: {sum(v > 1 for v in seen.values())}", flush=True)


if __name__ == "__main__":
    main()
