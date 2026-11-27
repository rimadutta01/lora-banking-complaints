"""Train DistilBERT on Banking77: --mode full (baseline) or --mode lora (improved)."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL = "distilbert-base-uncased"

def dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "lora"], required=True)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    a = p.parse_args()
    epochs = a.epochs or (3 if a.mode == "full" else 5)
    lr = a.lr or (5e-5 if a.mode == "full" else 1e-3)
    name = "full" if a.mode == "full" else f"lora_r{a.r}"

    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    if a.mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, target_modules=["q_lin", "v_lin"],
                         lora_dropout=0.1, task_type="SEQ_CLS")
        model = get_peft_model(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def metrics(ep):
        pred = ep.predictions.argmax(-1)
        return {"accuracy": accuracy_score(ep.label_ids, pred),
                "macro_f1": f1_score(ep.label_ids, pred, average="macro")}

    args = TrainingArguments(output_dir=f"/tmp/{name}", num_train_epochs=epochs,
                             learning_rate=lr, per_device_train_batch_size=32,
                             per_device_eval_batch_size=64, eval_strategy="no",
                             save_strategy="no", report_to="none", seed=42,
                             fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=enc["train"],
                      eval_dataset=enc["test"], data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    res = trainer.evaluate()
    save_dir = "models/full" if a.mode == "full" else f"models/{name}_adapter"
    model.save_pretrained(save_dir)
    json.dump(labels, open(os.path.join(save_dir, "labels.json"), "w"))

    # Latency (single complaint)
    model.eval(); dev = next(model.parameters()).device
    x = tok("my upi payment failed", return_tensors="pt").to(dev)
    with torch.no_grad():
        for _ in range(5): model(**x)
        t = time.time()
        for _ in range(50): model(**x)
    latency_ms = (time.time() - t) / 50 * 1000

    # Extra Indian-style test set
    ind_acc = None
    if os.path.exists("src/indian_test.csv"):
        df = pd.read_csv("src/indian_test.csv")
        l2i = {l: i for i, l in enumerate(labels)}
        df = df[df.label.isin(l2i)]
        with torch.no_grad():
            preds = [model(**tok(t, return_tensors="pt").to(dev)).logits.argmax(-1).item() for t in df.text]
        ind_acc = float(np.mean([p == l2i[l] for p, l in zip(preds, df.label)]))

    out = {"model": name, "trainable_params": trainable, "total_params": total,
           "trainable_pct": round(100 * trainable / total, 3),
           "accuracy": round(res["eval_accuracy"], 4), "macro_f1": round(res["eval_macro_f1"], 4),
           "indian_test_acc": ind_acc, "train_time_s": round(train_time, 1),
           "peak_gpu_mem_mb": round(peak, 1), "saved_size_mb": round(dir_size_mb(save_dir), 2),
           "latency_ms": round(latency_ms, 2)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/{name}.json", "w"), indent=2)
    print(out)

if __name__ == "__main__":
    main()
