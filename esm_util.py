"""ESM-2 sequence embeddings, shared by scripts/esm_embed.py (bundle building) and predict.py (fallback).

A protein's vector is the mean of the last hidden layer over residues; sequences longer than the window are
split into consecutive chunks whose means are combined weighted by length; several FASTA records are averaged.
"""
import numpy as np, torch

DEFAULT_MODEL = "facebook/esm2_t30_150M_UR50D"


class ESM:
    def __init__(self, path=DEFAULT_MODEL, device="cpu", chunk=1000):
        from transformers import AutoTokenizer, EsmModel  # imported lazily: only needed for unknown proteins
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = EsmModel.from_pretrained(path).to(device).eval()
        self.device, self.chunk = device, chunk

    @torch.no_grad()
    def embed_seq(self, seq):
        seq = seq.strip().upper().replace("*", "")
        acc, n = 0, 0
        for i in range(0, len(seq), self.chunk):
            t = self.tok(seq[i:i + self.chunk], return_tensors="pt").to(self.device)
            h = self.model(**t).last_hidden_state[0, 1:-1]  # drop BOS/EOS
            acc = acc + h.sum(0); n += h.shape[0]
        return (acc / n).float().cpu().numpy()

    def embed_records(self, seqs):
        return np.mean([self.embed_seq(s) for s in seqs], 0)
