"""Kaggle script kernel: generator-surrogate log-likelihood-ratio feature (S6E9 2nd-place recipe, GReaT-style).

Hypothesis: the competition rows were sampled from a language model fitted on the original dataset. A small LM
fine-tuned ONLY on the original 129,880 rows (label included as a field) approximates the generator's own
P(label | row). For every competition row we score
    LLR = log p(row | "satisfaction is satisfied") - log p(row | "satisfaction is neutral or dissatisfied")
averaged over K random column orders. Label-free w.r.t. competition labels -> safe as a feature for any model.
Outputs: llr_train.npy (train order), llr_test.npy (test order), llr_meta.json. No secrets needed.
"""
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "32"))
K_ORDERS = int(os.environ.get("K_ORDERS", "2"))
SCORE_BATCH = int(os.environ.get("SCORE_BATCH", "64"))
MAX_TRAIN_MIN = int(os.environ.get("MAX_TRAIN_MIN", "150"))  # cap fine-tuning wall time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40"], check=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

t0 = time.time()
dev = "cuda"
hits = glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)
D = os.path.dirname(hits[0])
train, test = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
orig_dir = os.path.dirname(glob.glob("/kaggle/input/**/airline-passenger-satisfaction/train.csv", recursive=True)[0])
orig = pd.concat([pd.read_csv(f"{orig_dir}/train.csv"), pd.read_csv(f"{orig_dir}/test.csv")], ignore_index=True)
orig = orig.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
COLS = [c for c in train.columns if c not in ("id", "satisfaction")]
LABEL_TXT = {1: "satisfied", 0: "neutral or dissatisfied"}
orig["satisfaction"] = (orig["satisfaction"] == "satisfied").astype(int)
print(f"orig {orig.shape}, train {train.shape}, test {test.shape}", flush=True)


def fmt(v):
    if isinstance(v, float):
        return str(int(v)) if v == v and float(v).is_integer() else ("nan" if v != v else f"{v:g}")
    return str(v)


def row_text(row, cols, label):
    body = ", ".join(f"{c} is {fmt(row[c])}" for c in cols)
    return f"satisfaction is {LABEL_TXT[label]}, {body}"


tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev)

# ---- fine-tune on the original rows (random column order per sample, label as the first field)
rng = random.Random(0)
orig_rows = orig.to_dict("records")
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
steps_total = EPOCHS * math.ceil(len(orig_rows) / BATCH)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps_total))
scaler = torch.cuda.amp.GradScaler()
model.train()
step, tstart = 0, time.time()
for ep in range(EPOCHS):
    rng.shuffle(orig_rows)
    for i in range(0, len(orig_rows), BATCH):
        batch = orig_rows[i:i + BATCH]
        texts = []
        for r in batch:
            cols = COLS[:]
            rng.shuffle(cols)
            texts.append(row_text(r, cols, int(r["satisfaction"])) + tok.eos_token)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**enc, labels=labels).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 500 == 0:
            print(f"ep {ep} step {step}/{steps_total} loss {loss.item():.4f} ({(time.time() - tstart) / 60:.1f} min)", flush=True)
        if (time.time() - tstart) / 60 > MAX_TRAIN_MIN:
            break
    if (time.time() - tstart) / 60 > MAX_TRAIN_MIN:
        print("time cap reached", flush=True)
        break
model.eval()
print(f"fine-tuned in {(time.time() - tstart) / 60:.1f} min", flush=True)
model.save_pretrained("ft_model")  # reusable for later scoring runs
tok.save_pretrained("ft_model")
del opt, scaler
torch.cuda.empty_cache()


@torch.no_grad()
def seq_logprob(texts):
    """Sum of token log-probs of each text (teacher-forced); fused cross-entropy avoids a full-vocab log_softmax copy."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc).logits[:, :-1]
    tgt = enc.input_ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                                            reduction="none").view(tgt.shape)
    mask = enc.attention_mask[:, 1:].float()
    return (-nll * mask).sum(1).cpu().numpy()


def score(df, name):
    rows = df.to_dict("records")
    out = np.zeros(len(rows), dtype=np.float32)
    orders = [random.Random(100 + k).sample(COLS, len(COLS)) for k in range(K_ORDERS)]
    ts = time.time()
    for i in range(0, len(rows), SCORE_BATCH):
        chunk = rows[i:i + SCORE_BATCH]
        acc = np.zeros(len(chunk), dtype=np.float32)
        for cols in orders:
            pos = seq_logprob([row_text(r, cols, 1) for r in chunk])
            neg = seq_logprob([row_text(r, cols, 0) for r in chunk])
            acc += pos - neg
        out[i:i + len(chunk)] = acc / K_ORDERS
        if (i // SCORE_BATCH) % 200 == 0:
            done = i + len(chunk)
            print(f"{name}: {done}/{len(rows)} ({(time.time() - ts) / 60:.1f} min)", flush=True)
    np.save(f"llr_{name}.npy", out)
    return out


llr_tr = score(train, "train")
auc = roc_auc_score(train["satisfaction"].astype(int).values, llr_tr)
print(f"LLR alone AUC on competition train: {auc:.6f}", flush=True)
llr_te = score(test, "test")
json.dump({"auc_train": auc, "epochs": EPOCHS, "k_orders": K_ORDERS, "minutes": (time.time() - t0) / 60},
          open("llr_meta.json", "w"))
print("done", (time.time() - t0) / 60, "min")
