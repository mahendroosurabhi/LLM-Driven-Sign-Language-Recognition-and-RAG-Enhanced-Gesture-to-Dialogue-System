import json

import numpy as np
import torch
import torch.nn as nn

N_FRAMES = 32


class SignGRU(nn.Module):
    def __init__(self, n_classes, in_dim=225, hidden=128):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True,bidirectional=True)
        self.drop = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden*2, n_classes)

    def forward(self, x):
        out, h = self.gru(x)
        pooled = out.mean(dim=1)          # or out.max(dim=1).values
        return self.fc(self.drop(pooled))


def normalize(seq):
    """seq: (32, 225) raw keypoints -> shoulders-centered, shoulder-width-scaled.
    Frames/points that were not detected (all zeros) stay zeros."""
    p = seq.reshape(len(seq), 75, 3).copy()
    for t in range(len(p)):
        l, r = p[t, 11], p[t, 12]
        if not (l.any() and r.any()):
            continue
        center = (l + r) / 2
        scale = np.linalg.norm(l[:2] - r[:2]) + 1e-6
        found = p[t].any(axis=1)
        p[t, found] = (p[t, found] - center) / scale
    return p.reshape(len(p), 225)

def resample(frames, n=N_FRAMES):
    """Any number of frame vectors -> exactly n evenly spaced ones (same idea as training)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[i] for i in idx])


class Recognizer:
    def __init__(self, model_path="best_model_aug.pt"):
        ckpt = torch.load(model_path, map_location="cpu")
        self.labels = ckpt["labels"]
        self.model = SignGRU(len(self.labels))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, seq_raw, k=3):
        x = torch.tensor(normalize(np.asarray(seq_raw)), dtype=torch.float32)[None]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        top = probs.topk(k)
        return [(self.labels[i], round(float(p), 3)) for p, i in zip(top.values, top.indices)]

    def predict_frames(self, frames, k=3):
        return self.predict(resample(frames), k)