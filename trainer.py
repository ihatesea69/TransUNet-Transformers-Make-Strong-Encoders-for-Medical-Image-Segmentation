import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def _format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def trainer_synapse(args, model, snapshot_path):
    from datasets.synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    device = next(model.parameters()).device
    checkpoint_dir = os.environ.get('TRANSUNET_CHECKPOINT_DIR', snapshot_path)
    mid_epoch_checkpoint_interval = int(os.environ.get('TRANSUNET_MID_EPOCH_SAVE_ITERS', '200'))
    iter_log_interval = int(os.environ.get('TRANSUNET_ITER_LOG_INTERVAL', '10'))
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               max_samples=args.max_train_samples,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
                             worker_init_fn=worker_init_fn)
    total_batches_per_epoch = len(trainloader)
    if args.n_gpu > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    start_epoch = 0
    start_batch = 0
    checkpoint_file = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        iter_num = checkpoint['iter_num']
        start_batch = checkpoint.get('batch_idx', 0)
        if start_batch > 0:
            logging.info(f"Resumed from checkpoint epoch {checkpoint['epoch']}, batch {start_batch}, iter {iter_num} (mid-epoch)")
        else:
            logging.info(f"Resumed from checkpoint epoch {checkpoint['epoch']}, iter {iter_num}")
            start_batch = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * total_batches_per_epoch
    logging.info("{} iterations per epoch. {} max iterations ".format(total_batches_per_epoch, max_iterations))
    if start_epoch > 0:
        logging.info(f"Resuming from epoch {start_epoch}/{max_epoch} (skipping {start_epoch} completed epochs)")
    best_performance = 0.0
    training_start_time = time.time()
    lr_ = base_lr
    for epoch_num in range(start_epoch, max_epoch):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_batches = 0
        model.train()

        print(f"\n{'='*70}", flush=True)
        print(f"  Epoch {epoch_num + 1}/{max_epoch}  |  Batches: {total_batches_per_epoch}  |  LR: {lr_:.6f}", flush=True)
        print(f"{'='*70}", flush=True)

        skip_batches = start_batch if epoch_num == start_epoch and start_batch > 0 else 0
        if skip_batches > 0:
            print(f"  Skipping {skip_batches} already-completed batches from mid-epoch checkpoint...", flush=True)

        for i_batch, sampled_batch in enumerate(trainloader):
            if i_batch < skip_batches:
                continue

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_ce_sum += loss_ce.item()
            epoch_batches += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if epoch_batches % iter_log_interval == 0:
                batch_elapsed = time.time() - epoch_start_time
                batch_speed = batch_elapsed / epoch_batches
                remaining_batches = total_batches_per_epoch - i_batch - 1
                batch_eta = remaining_batches * batch_speed
                print(
                    f"  [{i_batch + 1}/{total_batches_per_epoch}] "
                    f"loss={loss.item():.4f} ce={loss_ce.item():.4f} dice={loss_dice.item():.4f} "
                    f"lr={lr_:.6f} | "
                    f"{_format_duration(batch_elapsed)}<{_format_duration(batch_eta)}",
                    flush=True,
                )

            if iter_num % 20 == 0:
                vis_index = 1 if image_batch.size(0) > 1 else 0
                image = image_batch[vis_index, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[vis_index, ...] * 50, iter_num)
                labs = label_batch[vis_index, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if mid_epoch_checkpoint_interval > 0 and epoch_batches % mid_epoch_checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch_num,
                    'batch_idx': i_batch + 1,
                    'iter_num': iter_num,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        completed_epochs = epoch_num - start_epoch + 1
        remaining_epochs = max_epoch - epoch_num - 1
        avg_epoch_time = total_elapsed / completed_epochs
        eta_seconds = remaining_epochs * avg_epoch_time
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        avg_ce = epoch_ce_sum / max(epoch_batches, 1)
        epoch_summary = (
            f"\n>>> [Epoch {epoch_num + 1}/{max_epoch} DONE] "
            f"avg_loss={avg_loss:.4f} avg_ce={avg_ce:.4f} lr={lr_:.6f} | "
            f"epoch: {_format_duration(epoch_elapsed)} "
            f"total: {_format_duration(total_elapsed)} "
            f"ETA: {_format_duration(eta_seconds)} "
            f"({completed_epochs}/{max_epoch - start_epoch} epochs done)"
        )
        print(epoch_summary, flush=True)
        logging.info(epoch_summary)

        torch.save({
            'epoch': epoch_num,
            'batch_idx': 0,
            'iter_num': iter_num,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

    total_training_time = time.time() - training_start_time
    logging.info(f"Training completed in {_format_duration(total_training_time)}")
    writer.close()
    return "Training Finished!"
