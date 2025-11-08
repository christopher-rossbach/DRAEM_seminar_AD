import contextlib
import json
from io import StringIO
import time
import math
import numpy as np
import torch
import wandb
from test_DRAEM import evaluate_model_performance
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
import warnings

def init_wandb_run(
    run_name: str,
    project: str = "DRAEM_seminar_AD",
    config: dict = None,
    notes: str = "",
    tags: list = None,
    team: str = "team-cr",
    args: dict = None
):
    with contextlib.redirect_stdout(StringIO()):
        run = wandb.init(
            project=project,
            entity=team,
            name=run_name,
            config=config,
            notes=notes,
            tags=tags
        )
        return run


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:

        run_name = 'base_' + '_'.join(obj_names) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_ep' + str(args.epochs)

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()
        img_dim = 256

        train_dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[img_dim, img_dim])

        # Gradient accumulation: use micro-batch size 4, accumulate until args.bs samples then optimizer.step
        micro_batch_size = 4 if args.bs >= 4 else args.bs
        if args.bs % micro_batch_size != 0:
            raise ValueError(f"Batch size {args.bs} must be divisible by micro_batch_size {micro_batch_size}")

        train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=8)

        test_dataset = MVTecDRAEMTestDataset(args.data_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        config = vars(args)
        config.update({
            "train_dataset_size": len(train_dataset),
            "test_dataset_size": len(test_dataset),
            "obj_name": obj_name,
            "micro_batch_size": micro_batch_size,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "hostname": os.environ.get("HOSTNAME", "N/A"),
            "gpu_type": torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else "N/A",
        })
        # Merge default and user-supplied tags
        default_tags = ["train_draem"]
        extra_tags = []
        if getattr(args, "extra_tags", None):
            # Accept comma-separated string or repeated flags
            if isinstance(args.extra_tags, list):
                for item in args.extra_tags:
                    extra_tags.extend([t.strip() for t in item.split(",") if t.strip()])
            else:
                extra_tags = [t.strip() for t in str(args.extra_tags).split(",") if t.strip()]
        merged_tags = list(dict.fromkeys(extra_tags + default_tags))  # dedupe preserving order

        run = init_wandb_run(run_name=run_name, config=config, tags=merged_tags,)

        # Optional torch.compile (PyTorch 2.x). Can significantly speed up training for stable shapes.
        if getattr(args, "compile", False) and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                model_seg = torch.compile(model_seg)
            except Exception as _e:
                warnings.warn(f"torch.compile failed; continuing without it: {_e}")

        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            start_time = time.time()
            forward_times = []
            backward_times = []
            data_load_times = []
            l2_losses = []
            ssim_losses = []
            seg_losses = []
            total_batches = len(train_dataloader)
            micro_batches_per_batch = math.ceil(args.bs / micro_batch_size)
            optimizer.zero_grad(set_to_none=True)
            processed_in_group = 0  # samples accumulated toward next optimizer step
            
            for i_batch, sample_batched in enumerate(train_dataloader):
                data_start = time.time()
                gray_batch = sample_batched["image"].cuda(non_blocking=True)
                aug_gray_batch = sample_batched["augmented_image"].cuda(non_blocking=True)
                anomaly_mask = sample_batched["anomaly_mask"].cuda(non_blocking=True)
                data_load_times.append(time.time() - data_start)

                forward_start = time.time()
                with torch.cuda.amp.autocast(enabled=args.amp):
                    gray_rec = model(aug_gray_batch)
                    joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                    out_mask = model_seg(joined_in)
                    out_mask_sm = torch.softmax(out_mask, dim=1)

                # Compute losses in FP32 for numerical stability (especially SSIM)
                with torch.cuda.amp.autocast(enabled=False):
                    # Cast to float32 if AMP was used
                    gray_rec_loss = gray_rec.float() if args.amp else gray_rec
                    gray_batch_loss = gray_batch.float()
                    out_mask_sm_loss = out_mask_sm.float() if args.amp else out_mask_sm
                    
                    l2_loss = loss_l2(gray_rec_loss, gray_batch_loss)
                    ssim_loss = loss_ssim(gray_rec_loss, gray_batch_loss)
                    segment_loss = loss_focal(out_mask_sm_loss, anomaly_mask)
                    loss = l2_loss + ssim_loss + segment_loss
                forward_times.append(time.time() - forward_start)

                batch_size_actual = gray_batch.size(0)
                processed_in_group += batch_size_actual
                scale_factor = batch_size_actual / args.bs
                scaled_loss = loss * scale_factor
                if args.amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                l2_losses.append(l2_loss.item())
                ssim_losses.append(ssim_loss.item())
                seg_losses.append(segment_loss.item())

                last_batch = (i_batch + 1 == total_batches)
                if processed_in_group >= args.bs or last_batch:
                    backward_start = time.time()
                    if args.amp:
                        # Gradient clipping before optimizer step for stability
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(model_seg.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(model_seg.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    backward_times.append(time.time() - backward_start)
                    processed_in_group = 0
            start_evaluation_time = time.time()
            eval_results = None
            with torch.no_grad():
                eval_results = evaluate_model_performance(img_dim, model, model_seg, len(test_dataset), test_dataloader)


            auroc, ap, auroc_pixel, ap_pixel, display_images, display_gt_images, display_out_masks, display_in_masks = eval_results

            epoch_time = time.time() - start_time
            eval_time = time.time() - start_evaluation_time
            train_size = len(train_dataset)
            eval_size = len(test_dataset)

            # Aggregate average losses over micro-batches
            avg_l2 = float(np.mean(l2_losses)) if l2_losses else 0.0
            avg_ssim = float(np.mean(ssim_losses)) if ssim_losses else 0.0
            avg_seg = float(np.mean(seg_losses)) if seg_losses else 0.0

            log_data = {
                "train/l2_loss": avg_l2,
                "train/ssim_loss": avg_ssim,
                "train/segment_loss": avg_seg,
                "train/lr": get_lr(optimizer),
                "eval/auroc_image": auroc,
                "eval/ap_image": ap,
                "eval/auroc_pixel": auroc_pixel,
                "eval/ap_pixel": ap_pixel,
                "time/per_epoch": epoch_time,
                "time/data_load": sum(data_load_times) / train_size,
                "time/forward_passes": sum(forward_times) / train_size,
                "time/backwards_passes": sum(backward_times) / train_size,
                "time/per_sample": epoch_time / train_size,
                "time/evaluation": eval_time / eval_size,
            }
            run.log(log_data, step=epoch)
            print(json.dumps(log_data, indent=4))
            t_mask = out_mask_sm[:, 1:, :, :]

            if epoch % 20 == 0 or epoch == args.epochs - 1 or epoch < 5 or epoch == 10:
                img_grid_aug = torchvision.utils.make_grid(aug_gray_batch.detach().float().cpu(), normalize=True, scale_each=True)
                img_grid_target = torchvision.utils.make_grid(gray_batch.detach().float().cpu(), normalize=True, scale_each=True)
                img_grid_out = torchvision.utils.make_grid(gray_rec.detach().float().cpu(), normalize=True, scale_each=True)
                mask_grid_target = torchvision.utils.make_grid(anomaly_mask.detach().float().cpu(), normalize=False)
                mask_grid_out = torchvision.utils.make_grid(t_mask.detach().float().cpu(), normalize=False)

                run.log({
                    "images/batch_augmented": wandb.Image(img_grid_aug),
                    "images/batch_recon_target": wandb.Image(img_grid_target),
                    "images/batch_recon_out": wandb.Image(img_grid_out),
                    "images/mask_target": wandb.Image(mask_grid_target),
                    "images/mask_out": wandb.Image(mask_grid_out),
                }, step=epoch)

                eval_grid_recon_out = torchvision.utils.make_grid(display_images.float(), normalize=True, scale_each=True)
                eval_grid_recon_target = torchvision.utils.make_grid(display_gt_images.float(), normalize=True, scale_each=True)
                eval_grid_mask_out = torchvision.utils.make_grid(display_out_masks.float(), normalize=False)
                eval_grid_mask_target = torchvision.utils.make_grid(display_in_masks.float(), normalize=False)

                run.log({
                    "images/eval/recon_out_grid": wandb.Image(eval_grid_recon_out),
                    "images/eval/recon_target_grid": wandb.Image(eval_grid_recon_target),
                    "images/eval/mask_out_grid": wandb.Image(eval_grid_mask_out),
                    "images/eval/mask_target_grid": wandb.Image(eval_grid_mask_target),
                }, step=epoch)

                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
                torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))


            scheduler.step()
        run.finish()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--extra_tags', action='append', default=None, help='Additional W&B tags. Use multiple --extra_tags or a single comma-separated string.', required=False)
    parser.add_argument('--amp', action='store_true', default=False, help='Enable mixed precision (amp) for faster training and lower VRAM use.', required=False)
    parser.add_argument('--compile', action='store_true', default=False, help='Use torch.compile (PyTorch 2.x) to JIT-compile the model.', required=False)

    args = parser.parse_args()

    if args.extra_tags is None:
        args.extra_tags = ["no_tag"]


    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

