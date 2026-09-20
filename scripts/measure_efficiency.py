"""
Inference cost of the stage-2 skeleton model as a function of landmark count K.

Motivation: the paper reports subsets down to 1.8% of the skeleton. A reader may assume
that implies a proportional compute saving. It does not. The encoder flattens the K
landmarks of each frame and projects them to a fixed hidden size, so K affects ONLY the
input projection; the BERT stack always sees T=16 tokens of width 512 regardless of K.
This script quantifies that, so the claim is never made carelessly.

Architecture is reconstructed to match src/models/skeleton_encoders.py::BERTSkeletonEncoder
plus the classifier of src/models/lightning_asl.py. num_frames=16 is the resolved value of
the `video_mae` setting used in every stage-2 config.

Usage:  python scripts/measure_efficiency.py [--device cuda] [--out docs/EFFICIENCY.md]
"""
import argparse, json, time
import torch
import torch.nn as nn

K_VALUES = [543, 270, 100, 48, 24, 10]
DATASETS = {"AUTSL": 226, "ASL Citizen": 2731, "GSL": 310}
T, P, H, LAYERS, HEADS, INTER, FUSION = 16, 2, 512, 2, 8, 2048, 512


def build(K, num_classes):
    from transformers import BertConfig, BertModel
    cfg = BertConfig(hidden_size=H, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                     intermediate_size=INTER, max_position_embeddings=T,
                     vocab_size=1, type_vocab_size=1,
                     attention_dropout=0.2, hidden_dropout_prob=0.0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(K * P, H)
            self.norm = nn.LayerNorm(H)
            self.encoder = BertModel(cfg)
            self.head = nn.Linear(H, FUSION)
            self.classifier = nn.Sequential(
                nn.Linear(FUSION, FUSION), nn.GELU(), nn.Dropout(0.0),
                nn.Linear(FUSION, num_classes))

        def forward(self, x):                      # x: (B, T, K, P)
            B, t, j, p = x.shape
            x = self.proj(x.view(B, t, j * p))
            x = self.norm(x)
            x = self.encoder(inputs_embeds=x).last_hidden_state.mean(dim=1)
            return self.classifier(self.head(x))

    return Net()


def analytic_flops(K, num_classes):
    """Multiply-accumulates x2, batch size 1. Matches the forward above."""
    proj = K * P * H
    per_tok = 3 * H * H + 2 * T * H + H * H + 2 * H * INTER   # qkv, attn, out-proj, ffn
    bert = T * LAYERS * per_tok
    cls = H * FUSION + FUSION * FUSION + FUSION * num_classes
    return {"proj": 2 * proj, "bert": 2 * bert, "head_classifier": 2 * cls,
            "total": 2 * (proj + bert + cls)}


def time_forward(model, K, device, batch, reps=50, warmup=10):
    x = torch.randn(batch, T, K, P, device=device)
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return dt * 1000.0, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="docs/EFFICIENCY.md")
    args = ap.parse_args()
    dev = args.device
    print(f"device={dev}  torch={torch.__version__}")
    if dev == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    rows = []
    for name, C in DATASETS.items():
        for K in K_VALUES:
            m = build(K, C)
            tot = sum(p.numel() for p in m.parameters())
            pr = sum(p.numel() for p in m.proj.parameters())
            be = sum(p.numel() for p in m.encoder.parameters())
            cl = sum(p.numel() for p in m.head.parameters()) + \
                 sum(p.numel() for p in m.classifier.parameters())
            fl = analytic_flops(K, C)
            l1, pk1 = time_forward(m, K, dev, 1)
            l64, pk64 = time_forward(m, K, dev, 64)
            rows.append(dict(dataset=name, classes=C, K=K, params=tot, params_proj=pr,
                             params_bert=be, params_head=cl, flops=fl["total"],
                             flops_proj=fl["proj"], flops_bert=fl["bert"],
                             lat_b1_ms=l1, lat_b64_ms=l64, peak_b64_mib=pk64))
            print(f"{name:12s} K={K:4d}  params={tot/1e6:6.2f}M (proj {pr/1e6:5.3f}M)  "
                  f"GFLOPs={fl['total']/1e9:6.4f}  b1={l1:6.2f}ms  b64={l64:7.2f}ms")
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()

    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump({"device": dev, "rows": rows}, f, indent=1)

    with open(args.out, "w") as f:
        f.write(f"# Inference cost vs landmark count K\n\nDevice: `{dev}`")
        if dev == "cuda":
            f.write(f" ({torch.cuda.get_device_name(0)})")
        f.write(f", torch {torch.__version__}, T={T} frames, hidden={H}, {LAYERS} layers.\n\n")
        f.write("Only the input projection depends on K. The BERT stack always processes "
                f"T={T} tokens of width {H}.\n\n")
        for name, C in DATASETS.items():
            sub = [r for r in rows if r["dataset"] == name]
            base = sub[0]
            f.write(f"\n## {name} ({C} classes)\n\n")
            f.write("| K | params | of which proj | GFLOPs | % FLOPs in proj | "
                    "latency b=1 (ms) | latency b=64 (ms) | vs K=543 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in sub:
                f.write(f"| {r['K']} | {r['params']/1e6:.2f}M | {r['params_proj']/1e6:.3f}M | "
                        f"{r['flops']/1e9:.4f} | {100*r['flops_proj']/r['flops']:.1f}% | "
                        f"{r['lat_b1_ms']:.2f} | {r['lat_b64_ms']:.2f} | "
                        f"{100*r['flops']/base['flops']:.1f}% |\n")
    print(f"\nwrote {args.out} and {args.out.replace('.md','.json')}")


if __name__ == "__main__":
    main()
