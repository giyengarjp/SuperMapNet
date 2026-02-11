import os
import torch
import time
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.distributed import DistributedSampler
from mapmaster.models.network import MapMaster
from mapmaster.engine.core import MapMasterCli
from mapmaster.engine.experiment import BaseExp
from mapmaster.dataset.nuscenes_pivotnet import NuScenesMapDataset
from mapmaster.dataset.transform import Resize, Normalize, ToTensor_Pivot
from mapmaster.utils.misc import get_param_groups, is_distributed
from tools.evaluation.eval import compute_one_ap
from tools.evaluation.ap import instance_mask_ap as get_batch_ap
from mapmaster.dataset.visual import visual_map_pred
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

class EXPConfig:
    
    DATA_ROOT = './data/nuscenes/'
    IMAGE_SHAPE = (900, 1600)   # H, W

    map_conf = dict(
        version = 'v1.0-trainval',
        # version = 'v1.0-mini',
        # version = 'v1.0-mini',
        dataset_name="nuscenes",
        nusc_root='/workspace/nuscenes/',
        split_dir="assets/splits/nuscenes",
        num_classes=3,
        ego_size=(120, 30),
        map_region=(-60, 60, -15, 15),
        map_resolution=0.15,
        map_size=(800, 200),
        mask_key="instance_mask8",
        line_width=8,
        save_thickness=1,
    )

    pivot_conf = dict(
        max_pieces=(10, 2, 30),   # max num of pts in divider / ped / boundary]  #10, 2, 30
    )

    dataset_setup = dict(
        img_key_list=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        img_norm_cfg=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
        input_size=(896, 512),  #cv2  W, H
    )

    model_setup = dict(
        # image-branch
        im_backbone=dict(
            arch_name="swin_transformer",
            bkb_kwargs=dict(arch="tiny", 
                            shift_mode=0, 
                            out_indices=(2, 3), 
                            use_checkpoint=True,
                            pretrained='assets/weights/upernet_swin_tiny_patch4_window7_512x512.pth'),
            ret_layers=2,
            fpn_kwargs=None,
        ),
        bev_decoder=dict(
            arch_name="ipm_deformable_transformer",
            net_kwargs=dict(
                in_channels=[384, 768]  ,   #768
                src_shape=[(32, 336), (16, 168)],    #32 56*6, 16, 28*6
                tgt_shape=(100, 25),    
                d_model=256,
                n_heads=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                return_intermediate_dec=True,
                dec_n_points=8,
                enc_n_points=8,
                src_pos_encode="learned",
                tgt_pos_encode="learned",
                norm_layer=nn.SyncBatchNorm,
                use_checkpoint=False,
                use_projection=True,
                map_size=map_conf["map_size"],
                map_resolution=map_conf["map_resolution"],
                image_shape=(900, 1600),
                image_order=[2, 1, 0, 5, 4, 3]
            )
        ),
        lidar_encoder=dict(
            arch_name="pointpillar_encoder",
            net_kwargs=dict(
                 C=256, 
                 xbound=[-60.0, 60.0, 0.15], 
                 ybound=[-15.0, 15.0, 0.15], 
                 zbound=[-10.0, 10.0, 20.0], 
                 ppdim=4,
            )
        ),
        cross_encoder=dict(
            arch_name="CrossEncoder",
            net_kwargs=dict(
                 tgt_shape=(100, 25),
                 use_cross=True, 
                 num_heads=8, 
                 pos_emd=True, 
                 neck_dim=256,
                 cross_dim=256 
            )
        ),
        fusion_encoder=dict(
            # arch_name="ConcatBEV",  #ConcatBEV, BevFusionEncoder
            arch_name="BevFusionEncoder",  #ConcatBEV, BevFusionEncoder
            net_kwargs=dict(
                 features=512, 
            )
        ),
        ins_decoder=dict(
            arch_name=  "line_aware_decoder", #"point_element_decoder",  #"line_aware_decoder", #
            net_kwargs=dict(
                decoder_ids=[0, 1, 2, 3, 4, 5],
                in_channels=256,
                num_feature_levels=1,
                mask_classification=True,
                num_classes=1,
                hidden_dim=256,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=6,
                pre_norm=False,
                mask_dim=256,
                enforce_input_project=False,
                query_split=(20, 25, 15),   #20, 25, 15
                max_pieces=pivot_conf["max_pieces"], 
                dropout = 0.0,
            ),
        ),
        output_head=dict(
            arch_name="pivot_point_predictor",
            net_kwargs=dict(
                in_channel=256,
                num_queries=[20, 25, 15],   #20, 25, 15
                tgt_shape=map_conf['map_size'],
                max_pieces=pivot_conf["max_pieces"],
                bev_channels=256,
                ins_channel=64,
            )
        ),

        post_processor=dict(
            arch_name="pivot_post_processor",
            net_kwargs=dict(
                criterion_conf=dict(
                    weight_dict=dict(
                        sem_msk_loss=3,
                        ins_obj_loss=2, ins_msk_loss=5,
                        pts_loss=30, collinear_pts_loss=10, 
                        pt_logits_loss=2,
                    ),
                    decoder_weights=[0.4, 0.4, 0.4, 0.8, 1.2, 1.6]
                ),
                matcher_conf=dict(
                    cost_obj=2, cost_mask=5,
                    coe_endpts=5,
                    cost_pts=30,
                    mask_loss_conf=dict(
                        ce_weight=1,
                        dice_weight=1,
                    )
                ),
                pivot_conf=pivot_conf,
                map_conf=map_conf,
                sem_loss_conf=dict(
                    decoder_weights=[0.4, 0.8, 1.6, 2.4],
                    mask_loss_conf=dict(ce_weight=1, dice_weight=1)),
                no_object_coe=0.5,
                collinear_pts_coe=0.2,
                coe_endpts=5,
            )
        )
    )

    optimizer_setup = dict(
        base_lr=2e-4,
        wd=1e-4,
        backb_names=["backbone"],
        backb_lr=5e-5,
        extra_names=[ ],#'lidar_encoder'
        extra_lr=5e-5,
        freeze_names=[],
    )

    scheduler_setup = dict(
        gamma=0.9,
    )

    metric_setup = dict(
        map_resolution=map_conf["map_resolution"],
        iou_thicknesses=(1,),
        cd_thresholds=(0.2, 0.5, 1.0, 1.5, 5.0)
    )
    
    VAL_TXT = [
        "assets/splits/nuscenes/val.txt", 
    ]

import warnings

warnings.filterwarnings("ignore")

class Exp(BaseExp):
    def __init__(self, batch_size_per_device=1, total_devices=4, max_epoch=60, **kwargs):
        super(Exp, self).__init__(batch_size_per_device, total_devices, max_epoch)

        self.exp_config = EXPConfig()
        self.data_loader_workers = 1
        self.print_interval = 100
        self.dump_interval = 1
        self.eval_interval = 1
        self.seed = 0
        self.num_keep_latest_ckpt = 1
        self.ckpt_oss_save_dir = None
        self.enable_tensorboard = True
        self.max_line_count = 100

        #milestones = self.exp_config.scheduler_setup["milestones"]
        #self.exp_config.scheduler_setup["milestones"] = [int(x * max_epoch) for x in milestones]

        lr_ratio_dict = {32: 2, 16: 1.5, 8: 1, 4: 0.5, 2: 0.5, 1: 0.5}
        assert total_devices in lr_ratio_dict, "Please set normal devices!"
        for k in ['base_lr', 'backb_lr', 'extra_lr']:
            self.exp_config.optimizer_setup[k] = self.exp_config.optimizer_setup[k] * lr_ratio_dict[total_devices]
        self.evaluation_save_dir = None

    def _configure_model(self):
        model = MapMaster(self.exp_config.model_setup)
        return model

    def _configure_train_dataloader(self):
        from mapmaster.dataset.sampler import InfiniteSampler

        dataset_setup = self.exp_config.dataset_setup

        transform = Compose(
            [
                Resize(img_scale=dataset_setup["input_size"]),
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor_Pivot(),
            ]
        )

        train_set = NuScenesMapDataset(
            img_key_list=dataset_setup["img_key_list"],
            map_conf=self.exp_config.map_conf,
            point_conf = self.exp_config.pivot_conf,
            transforms=transform,
            data_split="train",
        )

        if is_distributed():
            sampler = InfiniteSampler(len(train_set), seed=self.seed if self.seed else 0)
        else:
            sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size_per_device,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=sampler is None,
            drop_last=True,
            sampler=sampler,
        )
        self.train_dataset_size = len(train_set)
        return train_loader

    def _configure_val_dataloader(self):

        dataset_setup = self.exp_config.dataset_setup

        transform = Compose(
            [
                Resize(img_scale=dataset_setup["input_size"]),
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor_Pivot(),
            ]
        )

        val_set = NuScenesMapDataset(
            img_key_list=dataset_setup["img_key_list"],
            map_conf=self.exp_config.map_conf,
            point_conf = self.exp_config.pivot_conf,
            transforms=transform,
            data_split="val",
        )

        if is_distributed():
            sampler = DistributedSampler(val_set, shuffle=False)
        else:
            sampler = None

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )

        self.val_dataset_size = len(val_set)
        return val_loader

    def _configure_test_dataloader(self):
        dataset_setup = self.exp_config.dataset_setup

        transform = Compose(
            [
                Resize(img_scale=dataset_setup["input_size"]),
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor_Pivot(),
            ]
        )

        test_set = NuScenesMapDataset(
            img_key_list=dataset_setup["img_key_list"],
            map_conf=self.exp_config.map_conf,
            point_conf = self.exp_config.pivot_conf,
            transforms=transform,
            data_split="val_sub",
            # data_split="test",
        )

        if is_distributed():
            sampler = DistributedSampler(test_set, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            pin_memory=True,
            num_workers=self.data_loader_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )

        self.test_dataset_size = len(test_set)
        print(f"✓ Test dataset size: {self.test_dataset_size}")
        print(f"✓ Test dataset size: {self.test_dataset_size}")
        return test_loader

    def _configure_optimizer(self):
        optimizer_setup = self.exp_config.optimizer_setup
        optimizer = AdamW(get_param_groups(self.model, optimizer_setup))
        return optimizer

    def _configure_lr_scheduler(self):
        scheduler_setup = self.exp_config.scheduler_setup
        iters_per_epoch = len(self.train_dataloader)
        scheduler = ExponentialLR(
            optimizer=self.optimizer,
            gamma=scheduler_setup["gamma"],
            last_epoch=-1,
        )
        return scheduler

    def training_step(self, batch):
        batch["images"] = batch["images"].float().cuda()
        batch["lidars"] = batch["lidars"].float().cuda()
        batch["lidar_mask"] = batch["lidar_mask"].float().cuda()
        #batch["targets"] = batch["targets"].float().cuda()
        #print(batch.keys())
        #for name, param in self.model.named_parameters():
        #    if param.grad is None:
        #        print(name)
        outputs = self.model(batch)
        return self.model.module.post_processor(outputs["outputs"], batch["targets"])

    def save_results(self, tokens, results, dt_masks, batch=None):
        """
        Save predictions to disk as .npz files
        
        Args:
            tokens: List of sample identifiers
            results: List of prediction results dicts
            dt_masks: List of predicted instance masks
            batch: Optional batch dict with extra scene information
        """
        
        # Create results directory on first call
        if self.evaluation_save_dir is None:
            # self.output_dir is set by BaseExp (e.g., outputs/pivotnet_nuscenes_swint/...)
            self.evaluation_save_dir = os.path.join(
                self.output_dir, 
                "evaluation", 
                "results"
            )
            os.makedirs(self.evaluation_save_dir, exist_ok=True)
            print(f"\n✓ Results will be saved to: {self.evaluation_save_dir}\n")
        
        # Save each sample as individual .npz file
        targets = batch['targets'] if batch is not None else None
        nusc_token = batch['extra_infos']['token'] if batch is not None else None
        ego_pose = batch['extra_infos']['ego_pose'] if batch is not None else None
        scene = batch['extra_infos']['scene'] if batch is not None else None
        frame_idx = batch['extra_infos']['frame_index'] if batch is not None else None

        for token, dt_res, dt_mask, target in zip(tokens, results, dt_masks, targets):
            save_path = os.path.join(self.evaluation_save_dir, f"{token}.npz")
            np.savez_compressed(
                save_path, 
                nusc_token=nusc_token,  # NuScenes token
                scene=scene,            # Scene name
                frame_idx=frame_idx,     # Frame index within scene
                ego_pose=ego_pose,      # Ego vehicle pose
                targets=targets,       # Ground truth targets (if available)
                dt_res=dt_res,        # Additional results (scores, lines, etc.)
                dt_mask=dt_mask,      # Instance mask predictions
            )
        # print(f"✓ Saved {len(tokens)} sample(s) to evaluation/results/")

    def save_visualizations(self, tokens, results, dt_masks, gt_masks=None, batch=None):
        """
        Save BEV visualizations of predictions and ground truth
        
        Args:
            tokens: List of sample identifiers
            results: List of prediction results dicts (contains pivot points/polylines)
            dt_masks: List of predicted instance masks (C, H, W)
            gt_masks: Optional ground truth masks (B, C, H, W) or (C, H, W)
            batch: Optional batch dict (contains ground truth polylines)
        """
        import cv2
        
        # Create visualization directory on first call
        vis_dir = os.path.join(
            os.path.dirname(self.evaluation_save_dir),
            "visualization"
        )
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get map configuration
        map_size = self.exp_config.map_conf["map_size"]  # (H, W)
        map_resolution = self.exp_config.map_conf["map_resolution"]
        
        # Color palette: class-specific colors
        class_colors = {
            0: (0, 0, 1),      # Blue for dividers
            1: (1, 0, 0),      # Red for pedestrians
            2: (0, 1, 0),      # Green for boundaries
        }
        
        # Visualize each sample
        for idx, token in enumerate(tokens):
            self._visualize_polylines(
                token, results[idx], batch, vis_dir, 
                map_size, map_resolution, class_colors)

    def _visualize_polylines(self, token, result, batch, vis_dir, map_size, map_resolution, class_colors, separate_plots=False):
        """
        Visualize predicted and ground truth polylines on BEV map
        
        Args:
            token: Sample identifier
            result: Prediction result dict with pivot points
            batch: Batch dict with ground truth polylines
            vis_dir: Directory to save visualization
            map_size: (H, W) of BEV map
            map_resolution: Resolution in meters per pixel
            class_colors: Dict mapping class_id to RGB colors
        """
        import cv2
        def _to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        # Draw ground truth polylines in dark colors
        points_dict = batch['targets']['points']
        vlen_dict   = batch['targets']['valid_len']

        # Map extents (for limits)
        map_size = batch['extra_infos']['map_size']
        L = float(_to_numpy(map_size[0]).reshape(-1)[0])
        Wm = float(_to_numpy(map_size[1]).reshape(-1)[0])

        fig, ax = plt.subplots(figsize=(16, 4), dpi=120)

        for cls in sorted(points_dict.keys()):
            pts = _to_numpy(points_dict[cls][0])        # (num_lines, max_pts, 2)
            vlen = _to_numpy(vlen_dict[cls][0]).astype(int)         # (num_lines,)
            
            base_color = class_colors.get(cls, (128, 128, 128))
            gt_color = tuple(int(c * 0.5) for c in base_color)  # Darken for GT
            
            for i in range(len(pts)):
                n = int(vlen[i])
                if n <= 0: 
                    continue
                xy = pts[i, :n, :]        # (n, 2)

                # polyline
                ax.plot(xy[:, 0]*L, xy[:, 1]*Wm, color=gt_color, linewidth=2, alpha=0.9)

                # point markers at vertices
                ax.scatter(xy[:, 0]*L, xy[:, 1]*Wm,
                        s=12, c=gt_color, edgecolors='black',
                        linewidths=0.8, alpha=0.9, zorder=3)
        
        # Ego vehicle marker at center
        ax.scatter([L//2], [Wm//2], s=120, c='black', edgecolors='black', zorder=5)
        ax.grid(True, alpha=0.3)
        
        vis_path = os.path.join(vis_dir, f"{token}_gt.png")
        fig.savefig(vis_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)
        # fig, ax = plt.subplots(figsize=(16, 4), dpi=120)

        # Draw predicted polylines in bright colors on top
        for i in range(len(result["map"])):
            
            points = result["map"][i]           # (n_pts, 2)
            class_id = result["pred_label"][i]
            
            base_color = class_colors.get(class_id-1, (128, 128, 128))
            if points is not None and len(points) >= 2:
                # Plot line
                ax.plot(points[:, 0]*map_resolution, points[:, 1]*map_resolution, 
                        color=base_color, linewidth=2, alpha=0.7)
                
                # Plot points as markers
                ax.scatter(points[:, 0]*map_resolution, points[:, 1]*map_resolution, 
                        color=base_color, s=12, edgecolors='black', 
                        linewidths=0.5, alpha=0.7, zorder=3)
        
        # Ego vehicle marker at center
        # ax.scatter([L//2], [Wm//2], s=120, c='black', edgecolors='black', zorder=5)
        # ax.grid(True, alpha=0.3)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"{token}_predictions.png")
        fig.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    def test_step(self, batch, step, ap_matrix, ap_count_matrix):
        """
        Evaluation step - compute AP metrics AND save predictions
        
        Args:
            batch: Input batch dictionary
            step: Current sample index
            ap_matrix: Array to accumulate AP scores
            ap_count_matrix: Array to track count
        
        Returns:
            Updated ap_matrix and ap_count_matrix
        """
        
        with torch.no_grad():
            # Prepare batch
            # Prepare batch
            batch["images"] = batch["images"].float().cuda()
            batch["lidars"] = batch["lidars"].float().cuda()
            batch["lidar_mask"] = batch["lidar_mask"].float().cuda()

            # Forward pass

            # Forward pass
            outputs = self.model(batch)
            results, dt_masks = self.model.module.post_processor(outputs["outputs"])
            
            # Extract sample tokens/identifiers
            tokens = batch.get('token', [f"sample_{step}_{i}" for i in range(len(dt_masks))])
            
            # ===== SAVE RESULTS and VISUALIZATIONS =====
            self.save_results(tokens, results, dt_masks, batch=batch)
            self.save_visualizations(
                tokens, results, dt_masks, 
                batch.get('targets', {}).get('masks', None),
                batch=batch)
            # ================================
            
            # Compute AP metrics
            map_resolution = (0.15, 0.15)
            # Extract sample tokens/identifiers
            tokens = batch.get('token', [f"sample_{step}_{i}" for i in range(len(dt_masks))])
            
            # ===== SAVE RESULTS and VISUALIZATIONS =====
            self.save_results(tokens, results, dt_masks, batch=batch)
            self.save_visualizations(
                tokens, results, dt_masks, 
                batch.get('targets', {}).get('masks', None),
                batch=batch)
            # ================================
            
            # Compute AP metrics
            map_resolution = (0.15, 0.15)
            SAMPLED_RECALLS = torch.linspace(0.1, 1, 10).cuda()
            max_line_count = 100
            max_line_count = 100
            THRESHOLDS = [0.2, 0.5, 1.0, 1.5]
            
            
            dt_masks = np.asarray(dt_masks)
            dt_scores = results[0]["confidence_level"]
            dt_scores = np.array(
                list(dt_scores) + [-1] * (max_line_count - len(dt_scores))
            )
            dt_scores = np.array(
                list(dt_scores) + [-1] * (max_line_count - len(dt_scores))
            )
            
            # Update AP matrices
            # Update AP matrices
            ap_matrix, ap_count_matrix = get_batch_ap(
                ap_matrix.cuda(),
                ap_count_matrix.cuda(),
                torch.from_numpy(dt_masks).cuda(),
                torch.from_numpy(dt_masks).cuda(),
                batch['targets']['masks'].cuda(),
                *map_resolution,
                torch.from_numpy(np.array(dt_scores)).unsqueeze(0).cuda(),
                THRESHOLDS,
                SAMPLED_RECALLS,
            )
            
            # Save accumulated AP matrices to disk
            metrics_dir = os.path.dirname(self.evaluation_save_dir)
            metrics_path = os.path.join(metrics_dir, "metrics.npz")
            np.savez_compressed(
                metrics_path,
                ap_matrix=ap_matrix.cpu().numpy(),
                ap_count_matrix=ap_count_matrix.cpu().numpy()
            )
            # Save accumulated AP matrices to disk
            metrics_dir = os.path.dirname(self.evaluation_save_dir)
            metrics_path = os.path.join(metrics_dir, "metrics.npz")
            np.savez_compressed(
                metrics_path,
                ap_matrix=ap_matrix.cpu().numpy(),
                ap_count_matrix=ap_count_matrix.cpu().numpy()
            )

        return ap_matrix, ap_count_matrix

if __name__ == "__main__":
    MapMasterCli(Exp).run()