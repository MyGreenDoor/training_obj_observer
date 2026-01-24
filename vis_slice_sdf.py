import json
import numpy as np
import matplotlib.pyplot as plt

def load_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    sdf = z["sdf"]
    meta = json.loads(z["meta"].tobytes().decode("utf-8"))
    print("signed_policy:", meta.get("signed_policy"))
    print("signed_used:", meta.get("signed_used"))
    print("sign_convention:", meta.get("sign_convention"))
    print("watertight:", meta.get("watertight_report"))

    print("any sdf<0:", np.any(sdf < 0))
    print("any signbit:", np.any(np.signbit(sdf)))  # -0.0 も拾う
    return sdf, meta

def quick_stats(sdf: np.ndarray):
    s = sdf
    near0 = np.mean(np.abs(s) < (np.std(s) * 1e-3 + 1e-4))
    neg = np.mean(s < 0)
    pos = np.mean(s > 0)
    print(f"shape={s.shape} dtype={s.dtype}")
    print(f"min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} std={s.std():.6g}")
    print(f"neg_ratio={neg:.3f} pos_ratio={pos:.3f} near0_ratio≈{near0:.3f}")

def show_slices(sdf: np.ndarray, title: str = ""):
    r = sdf.shape[0]
    cx = r // 2
    cy = r // 2
    cz = r // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(sdf[cx, :, :], origin="lower")
    axs[0].set_title("x mid")
    axs[1].imshow(sdf[:, cy, :], origin="lower")
    axs[1].set_title("y mid")
    axs[2].imshow(sdf[:, :, cz], origin="lower")
    axs[2].set_title("z mid")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

npz_path = r"/mnt/ssd2tb/new_format_sdf/original/SBS96/sdf/0011000200000.npz"
sdf, meta = load_npz(npz_path)
quick_stats(sdf)
show_slices(sdf, title=npz_path)
