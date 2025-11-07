import contextlib
import json
from io import StringIO
import time
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

def init_wandb_run(
    run_name: str,
    project: str = "DRAEM_seminar_AD",
    config: dict = None,
    notes: str = "",
    tags: list = None,
    team: str = "team-cr",
    args: dict = None
):
    config = args
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

        run_name = 'DRAEM_base_' + '_'.join(obj_names) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_ep' + str(args.epochs)

    # Images will be logged to Weights & Biases instead of TensorBoard.

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()
        img_dim = 256

        train_dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[img_dim, img_dim])

        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)

        test_dataset = MVTecDRAEMTestDataset(args.data_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        config = vars(args)
        config.update({"train_dataset_size": len(train_dataset), "test_dataset_size": len(test_dataset), })
        run = init_wandb_run( run_name=run_name, config=config, tags=["baseline"],)

        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            start_time = time.time()
            forward_times = []
            backward_times = []
            data_load_times = []
            
            batch_iter = iter(train_dataloader)
            for i_batch in range(len(train_dataloader)):
                data_start = time.time()
                sample_batched = next(batch_iter)
                data_load_times.append(time.time() - data_start)
                
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                forward_start = time.time()
                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss
                forward_times.append(time.time() - forward_start)

                backward_start = time.time()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                backward_times.append(time.time() - backward_start)
                

            start_evaluation_time = time.time()
            eval_results = None
            with torch.no_grad():
                eval_results = evaluate_model_performance(img_dim, model, model_seg, len(test_dataset), test_dataloader)


            auroc, ap, auroc_pixel, ap_pixel, display_images, display_gt_images, display_out_masks, display_in_masks = eval_results

            epoch_time = time.time() - start_time
            train_size = len(train_dataloader)
            eval_size = len(test_dataloader)

            log_data = {
                "train/l2_loss": l2_loss.item(),
                "train/ssim_loss": ssim_loss.item(),
                "train/segment_loss": segment_loss.item(),
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
                "time/evaluation": (time.time() - start_evaluation_time) / eval_size,
            }
            run.log(log_data, step=epoch)
            print(json.dumps(log_data, indent=4))
            t_mask = out_mask_sm[:, 1:, :, :]

            if epoch % 20 == 0 or epoch == args.epochs - 1 or epoch < 5 or epoch == 10:
                img_grid_aug = torchvision.utils.make_grid(aug_gray_batch.detach().cpu(), normalize=True, scale_each=True)
                img_grid_target = torchvision.utils.make_grid(gray_batch.detach().cpu(), normalize=True, scale_each=True)
                img_grid_out = torchvision.utils.make_grid(gray_rec.detach().cpu(), normalize=True, scale_each=True)
                mask_grid_target = torchvision.utils.make_grid(anomaly_mask.detach().cpu(), normalize=False)
                mask_grid_out = torchvision.utils.make_grid(t_mask.detach().cpu(), normalize=False)

                run.log({
                    "images/batch_augmented": wandb.Image(img_grid_aug),
                    "images/batch_recon_target": wandb.Image(img_grid_target),
                    "images/batch_recon_out": wandb.Image(img_grid_out),
                    "images/mask_target": wandb.Image(mask_grid_target),
                    "images/mask_out": wandb.Image(mask_grid_out),
                }, step=epoch)

                eval_grid_recon_out = torchvision.utils.make_grid(display_images, normalize=True, scale_each=True)
                eval_grid_recon_target = torchvision.utils.make_grid(display_gt_images, normalize=True, scale_each=True)
                eval_grid_mask_out = torchvision.utils.make_grid(display_out_masks, normalize=False)
                eval_grid_mask_target = torchvision.utils.make_grid(display_in_masks, normalize=False)

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

    args = parser.parse_args()

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

