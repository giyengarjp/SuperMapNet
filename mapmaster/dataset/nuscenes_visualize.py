"""
✓ Test dataset size: 100
dict_keys(['images', 'targets', 'lidars', 'lidar_mask', 'extra_infos', 'extrinsic', 'intrinsic'])
images: 
torch.Size([1, 6, 3, 512, 896])
targets: 
        masks: torch.Size([1, 3, 800, 200])
        points (dict):
                0: torch.Size([1, 3, 10, 2])
                1: torch.Size([1, 6, 2, 2])
                2: torch.Size([1, 4, 30, 2])
        valid_len (dict):
                0: torch.Size([1, 3])
                1: torch.Size([1, 6])
                2: torch.Size([1, 4])
lidars: 
torch.Size([1, 50000, 5])
lidar_mask: 
torch.Size([1, 50000])
extra_infos: 
        token: ['fd8420396768425eabec9bdddf7e64b6']
        map_size: [tensor([120]), tensor([30])]
        scale_factor: tensor([[1., 1., 1., 1.]])
        img_norm_cfg: {'mean': tensor([[0.4850, 0.4560, 0.4060]]), 'std': tensor([[0.2290, 0.2240, 0.2250]]), 'to_rgb': tensor([True])}
extrinsic: 
torch.Size([1, 6, 4, 4])
intrinsic: 
torch.Size([1, 6, 3, 3])
"""

"""
# Usage (example):
# Suppose `item` is what your DataLoader returns (batch size 1):
# item = {
#   'images': torch.Tensor[1,6,3,H,W],
#   'targets': {
#       'masks': torch.Tensor[1,3,800,200],
#       'points': {0: Tensor[1,N0,M0,2], 1: Tensor[1,N1,M1,2], 2: Tensor[1,N2,M2,2]},
#       'valid_len': {0: Tensor[1,N0], 1: Tensor[1,N1], 2: Tensor[1,N2]}
#   },
#   'extra_infos': {
#       'map_size': [tensor([120]), tensor([30])],
#       'img_norm_cfg': {'mean': tensor([[0.4850, 0.4560, 0.4060]]),
#                        'std': tensor([[0.2290, 0.2240, 0.2250]]),
#                        'to_rgb': tensor([True])}
#   }
# }
# paths = visualize_sample(item, out_dir="viz_out")
# print(paths)
"""

# viz_supermapnet.py
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
from PIL import Image
import re
from pathlib import Path

# ----------------------------
# Helpers
# ----------------------------
def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def inverse_normalize(img_chw, mean, std):
    """
    img_chw: (3, H, W) torch.Tensor or np.ndarray, float (normalized)
    mean/std: (3,) np arrays
    Returns uint8 RGB image (H, W, 3).
    """
    if isinstance(img_chw, torch.Tensor):
        img = img_chw.detach().cpu().numpy()
    else:
        img = np.asarray(img_chw)
    # (3,H,W) -> (H,W,3)
    img = np.transpose(img, (1,2,0))
    img = img * std.reshape(1,1,3) + mean.reshape(1,1,3)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    return img

def get_img_norm_params(img_norm_cfg):
    mean = _to_numpy(img_norm_cfg['mean']).reshape(-1)  # (3,)
    std  = _to_numpy(img_norm_cfg['std']).reshape(-1)   # (3,)
    # to_rgb is informational for upstream transforms; here we assume tensor is already RGB
    return mean, std

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ----------------------------
# 1) Camera grid (3x2)
# ----------------------------
def visualize_camera_grid(batch, out_path="images_grid.png",
                          camera_names=None, dpi=150, fontsize=10):
    """
    batch: dict with 'images' and 'extra_infos'
    camera_names: list of 6 names in the same order as images[:, i]
    """
    imgs = batch['images']            # [1, 6, 3, H, W]
    img_norm_cfg = batch['extra_infos']['img_norm_cfg']
    mean, std = get_img_norm_params(img_norm_cfg)

    imgs = imgs[0]  # [6, 3, H, W]
    B = imgs.shape[0]
    assert B == 6, f"Expected 6 cameras, got {B}"

    if camera_names is None:
        camera_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

    # Custom layout: FRONT on top, side cameras in middle, BACK on bottom
    fig = plt.figure(figsize=(10, 14), dpi=dpi)
    
    # Top: FRONT (index 1)
    ax_front = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    img = inverse_normalize(imgs[1], mean, std)
    ax_front.imshow(img)
    ax_front.set_axis_off()
    ax_front.text(8, 18, camera_names[1],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # Middle: FRONT_LEFT (index 0), FRONT_RIGHT (index 2)
    ax_fl = plt.subplot2grid((4, 2), (1, 0))
    img = inverse_normalize(imgs[0], mean, std)
    ax_fl.imshow(img)
    ax_fl.set_axis_off()
    ax_fl.text(8, 18, camera_names[0],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    ax_fr = plt.subplot2grid((4, 2), (1, 1))
    img = inverse_normalize(imgs[2], mean, std)
    ax_fr.imshow(img)
    ax_fr.set_axis_off()
    ax_fr.text(8, 18, camera_names[2],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # Middle: BACK_LEFT (index 3), BACK_RIGHT (index 5)
    ax_bl = plt.subplot2grid((4, 2), (2, 0))
    img = inverse_normalize(imgs[3], mean, std)
    ax_bl.imshow(img)
    ax_bl.set_axis_off()
    ax_bl.text(8, 18, camera_names[3],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    ax_br = plt.subplot2grid((4, 2), (2, 1))
    img = inverse_normalize(imgs[5], mean, std)
    ax_br.imshow(img)
    ax_br.set_axis_off()
    ax_br.text(8, 18, camera_names[5],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # Bottom: BACK (index 4)
    ax_back = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    img = inverse_normalize(imgs[4], mean, std)
    ax_back.imshow(img)
    ax_back.set_axis_off()
    ax_back.text(8, 18, camera_names[4],
            color='white', fontsize=fontsize, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path

# ----------------------------
# 2) BEV raster (classes 0/1/2 -> blue/red/green)
# ----------------------------

def visualize_bev_raster(batch, out_path="bev_raster.png"):
    """
    Directly visualize the BEV mask as an RGB numpy array.
    classes:
      0 -> blue
      1 -> red
      2 -> green
    """
    import numpy as np
    import matplotlib.pyplot as plt

    masks = batch['targets']['masks'][0]   # [3, H, W]
    masks = (masks > 0.5).cpu().numpy().astype(np.uint8)

    # Build RGB image (H, W, 3)
    rgb = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    rgb[..., 2] = masks[0] * 255  # blue
    rgb[..., 0] = masks[1] * 255  # red
    rgb[..., 1] = masks[2] * 255  # green

    plt.figure()#(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    return out_path

# ----------------------------
# 3) BEV vectorized polylines (ego frame)
# ----------------------------
def visualize_bev_vectors(batch, out_path="bev_vectors.png", dpi=150, lw=2.0, ego_size=2.0):
    """
    targets['points']: dict[class_id] -> [1, num_lines, max_pts, 2]
    targets['valid_len']: dict[class_id] -> [1, num_lines]  (number of valid pts per line)
    Colors: class 0 = blue, 1 = red, 2 = green
    Ego at (0,0). Axes in meters, equal aspect.
    """
    points_dict = batch['targets']['points']
    vlen_dict   = batch['targets']['valid_len']

    # Map extents (for limits)
    map_size = batch['extra_infos']['map_size']
    L = float(_to_numpy(map_size[0]).reshape(-1)[0])
    Wm = float(_to_numpy(map_size[1]).reshape(-1)[0])

    COLOR_MAP = {0: 'blue', 1: 'red', 2: 'green'}
    CLASS_MAP = {0: 'divider', 1: 'crosswalk', 2: 'boundary'}
    fig, ax = plt.subplots(figsize=(16, 4), dpi=120)

    for cls in sorted(points_dict.keys()):
        pts = _to_numpy(points_dict[cls][0])        # (num_lines, max_pts, 2)
        vlen = _to_numpy(vlen_dict[cls][0]).astype(int)         # (num_lines,)
        color = COLOR_MAP.get(cls, 'black')

        for i in range(len(pts)):
            n = int(vlen[i])
            if n <= 0: 
                continue
            xy = pts[i, :n, :]        # (n, 2)

            # polyline
            ax.plot(xy[:, 0]*L, xy[:, 1]*Wm, color=color, linewidth=2, alpha=0.9)

            # point markers at vertices
            ax.scatter(xy[:, 0]*L, xy[:, 1]*Wm,
                       s=12, c=color, edgecolors='black',
                       linewidths=0.8, alpha=0.9, zorder=3)

    # Ego vehicle marker at center
    ax.scatter([L//2], [Wm//2], s=40, c='white', edgecolors='black', zorder=5)
    ax.grid(True, alpha=0.3)
    
    # Legend (proxy lines)
    # from matplotlib.lines import Line2D
    # handles = [Line2D([0], [0], color=COLOR_MAP.get(k, 'black'), lw=2, label=f'{CLASS_MAP[k]}')
    #            for k in sorted(points_dict.keys())]
    # ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    
    return out_path

def visualize_sample(batch,
                     out_dir="viz_out",
                     camera_names=None):
    ensure_dir(out_dir)
    paths = {}
    paths['images_grid'] = visualize_camera_grid(batch, os.path.join(out_dir, "images_grid.png"), camera_names=camera_names)
    paths['bev_raster']  = visualize_bev_raster(batch, os.path.join(out_dir, "bev_raster.png"))
    paths['bev_vectors'] = visualize_bev_vectors(batch, os.path.join(out_dir, "bev_vectors.png"))
    return paths

def natural_sort_key(s):
    """Sort filenames naturally (e.g., 1, 2, 10 instead of 1, 10, 2)"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def create_gifs(viz_folder, output_dir="gifs"):
    """
    Create GIFs from visualization images.
    Assumes folder structure with gt_* and pred_* prefixed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files
    all_images = glob.glob(os.path.join(viz_folder, "*.png"))
    
    # Separate GT and predictions
    gt_images = sorted([f for f in all_images if '_gt' in os.path.basename(f)], 
                       key=lambda x: natural_sort_key(os.path.basename(x)))
    pred_images = sorted([f for f in all_images if '_predictions' in os.path.basename(f)], 
                         key=lambda x: natural_sort_key(os.path.basename(x)))
    
    # Create GT GIF
    if gt_images:
        print(f"Creating GT GIF with {len(gt_images)} frames...")
        frames = [Image.open(img).convert('RGB') for img in gt_images]
        frames[0].save(
            os.path.join(output_dir, 'ground_truth.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        # print(f"✓ Saved: {os.path.join(output_dir, 'ground_truth.gif')}")
    
    # Create predictions GIF
    if pred_images:
        print(f"Creating predictions GIF with {len(pred_images)} frames...")
        frames = [Image.open(img).convert('RGB') for img in pred_images]
        frames[0].save(
            os.path.join(output_dir, 'predictions.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        # print(f"✓ Saved: {os.path.join(output_dir, 'predictions.gif')}")
    
    # Create combined side-by-side GIF
    if gt_images and pred_images:
        print("Creating side-by-side comparison GIF...")
        min_len = min(len(gt_images), len(pred_images))
        frames = []
        
        for i in range(min_len):
            gt_img = Image.open(gt_images[i]).convert('RGB')
            pred_img = Image.open(pred_images[i]).convert('RGB')
            
            # Resize to same height
            w = min(gt_img.width, pred_img.width)
            gt_img = gt_img.resize((w, int(gt_img.height * w/gt_img.width)))
            pred_img = pred_img.resize((w, int(pred_img.height * w/pred_img.width)))
            
            # Concatenate horizontally
            combined = Image.new('RGB', (w, gt_img.height + pred_img.height))
            combined.paste(gt_img, (0, 0))
            combined.paste(pred_img, (0, gt_img.height))
            frames.append(combined)
        
        frames[0].save(
            os.path.join(output_dir, 'comparison.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        # print(f"✓ Saved: {os.path.join(output_dir, 'comparison.gif')}")

if __name__ == "__main__":
    
    # viz_folder = "/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint_dense/latest/evaluation/visualization/"
    # gif_folder = "/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint_dense/latest/evaluation/gifs/"
    # create_gifs(viz_folder, gif_folder)
    # print("Done! Check the 'gifs' folder.")

    def load_npz_verbose(path):
        
        data = np.load(path, allow_pickle=True)   # allow_pickle=True is REQUIRED for dicts
        print(f"---- Keys in {path} ----")
        print(list(data.keys()))
        print()

        for key in data.keys():
            print(f"===== {key} =====")

            obj = data[key]

            # Case 1: object arrays (e.g., list of polylines, dictionaries)
            if obj.dtype == object:
                obj = obj.item() if obj.size == 1 else obj
                print(type(obj))
                # print(obj)
                print(obj.keys() if isinstance(obj, dict) else f"len: {len(obj)}")
                print()

            # Case 2: numpy ndarrays
            else:
                print("type:", type(obj))
                print("shape:", obj.shape)
                # print(obj)
                print()

        print("---- Done ----")

    results_dir = Path("/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint_dense/latest/evaluation/results/")
    for i in range(100):  # 0..99 inclusive
        fname = f"sample_{i}_0.npz"
        result_name = results_dir / fname
        if not result_name.exists():
            print(f"Missing: {result_name}")    
        load_npz_verbose(result_name)
        break

    """
    # Base paths
    dense_dir = Path("/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint_dense/latest/evaluation/visualization")
    base_dir  = Path("/workspace/SuperMapNet/outputs/pivotnet_nuscenes_swint/latest/evaluation/visualization")
    out_dir   = Path("/workspace/SuperMapNet/viz_outputs/compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    def stack_vertical_centered(img1: Image.Image, img2: Image.Image) -> Image.Image:
        '''Stack two images vertically on a canvas whose width is max(w1, w2), centered horizontally'''
        w1, h1 = img1.size
        w2, h2 = img2.size
        W = max(w1, w2)
        H = h1 + h2

        # Use a transparent canvas if images have alpha; otherwise RGB white background
        mode = 'RGBA' if ('A' in img1.getbands() or 'A' in img2.getbands()) else 'RGB'
        bg = (255, 255, 255, 0) if mode == 'RGBA' else (255, 255, 255)
        canvas = Image.new(mode, (W, H), bg)

        x1 = (W - w1) // 2
        x2 = (W - w2) // 2
        canvas.paste(img1, (x1, 0))
        canvas.paste(img2, (x2, h1))
        return canvas

    missing = []
    made = 0

    for i in range(100):  # 0..99 inclusive
        fname = f"sample_{i}_0_predictions.png"
        p_dense = dense_dir / fname
        p_base  = base_dir  / fname

        if not p_dense.exists() or not p_base.exists():
            missing.append((i, not p_dense.exists(), not p_base.exists()))
            continue

        # Open images
        with Image.open(p_dense) as im_dense, Image.open(p_base) as im_base:
            # Convert to consistent mode to avoid paste issues
            if im_dense.mode not in ("RGB", "RGBA"):
                im_dense = im_dense.convert("RGBA")
            if im_base.mode not in ("RGB", "RGBA"):
                im_base = im_base.convert("RGBA")

            stacked = stack_vertical_centered(im_dense, im_base)

        # Save with a clear name
        out_path = out_dir / f"compare_i_{i}.png"
        stacked.save(out_path)
        made += 1

    print(f"Created {made} stacked images in: {out_dir}")
    if missing:
        print("Missing pairs (i, dense_missing, base_missing):")
        for entry in missing:
            print(entry)
    """