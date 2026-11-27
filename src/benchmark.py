"""Merge results/*.json into metrics.csv and draw comparison charts."""
import glob, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

df = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob("results/*.json"))])
df.to_csv("results/metrics.csv", index=False)
print(df.to_string(index=False))

for col, title, fn in [("accuracy", "Test accuracy", "accuracy_comparison.png"),
                       ("trainable_params", "Trainable parameters", "params_comparison.png"),
                       ("saved_size_mb", "Saved checkpoint size (MB)", "size_comparison.png"),
                       ("train_time_s", "Training time (s)", "time_comparison.png"),
                       ("peak_gpu_mem_mb", "Peak GPU memory (MB)", "memory_comparison.png")]:
    ax = df.plot.bar(x="model", y=col, legend=False, rot=0, figsize=(6, 4), title=title)
    for c in ax.containers: ax.bar_label(c, fontsize=8)
    plt.tight_layout(); plt.savefig(f"results/{fn}", dpi=150); plt.close()
