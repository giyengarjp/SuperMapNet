import os
import matplotlib.pyplot as plt

log_path = "/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint/latest/train.log"
out_dir = os.path.dirname(log_path)

keys = [
    "loss",
    "ins_msk_loss",
    "ins_obj_loss",
    "pts_loss",
    "collinear_pts_loss",
    "pt_logits_loss",
    "sem_msk_loss"
]

values = {k: [] for k in keys}
steps = []

with open(log_path, "r") as f:
    for line in f:
        if " loss=" not in line:
            continue

        parts = line.strip().split()
        record = {}

        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k in keys:
                try:
                    record[k] = float(v)
                except:
                    pass

        if "loss" in record:
            steps.append(len(steps))
            for k in keys:
                values[k].append(record.get(k, None))

# -----------------------------
# Plot main loss
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(steps, values["loss"], label="loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Main Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss.png"))
plt.close()

# -----------------------------
# Plot other loss components
# -----------------------------
plt.figure(figsize=(12, 6))
for k in keys:
    if k == "loss":
        continue
    plt.plot(steps, values[k], label=k)

plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Loss Components")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss_components.png"))
plt.close()

print("Saved:\n", 
      os.path.join(out_dir, "loss.png"), "\n",
      os.path.join(out_dir, "loss_components.png"))
