import math
from typing import Iterable

import torch
import torch.nn as nn
import utils.utils as utils
import torch.nn.functional as F


def get_img_and_label(epoch, tokenizer, batch, device):
    # (N, 3, 224, 224); (N, 3, 112, 112); (N, 14, 14)
    mask_weights = None
    if epoch >= 100:
        images, token_images, bool_masked_pos, mask_weights = batch
        mask_weights = mask_weights.to(device, non_blocking=True)
    else:
        images, token_images, bool_masked_pos = batch
    token_images = token_images.to(device, non_blocking=True)
    images = images.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

    with torch.no_grad():
        # get visual tokens, (N, 196)
        input_ids = tokenizer.get_codebook_indices(token_images).flatten(1)
        bool_masked_pos = bool_masked_pos.to(torch.bool).cpu()
        labels = input_ids[bool_masked_pos]
    return images, labels, bool_masked_pos, mask_weights


def updat_sampling_weights(transform, target, uncertainty, bool_masked_pos):
    # updating sampling weights
    class_name = transform.target_to_class_name[target.item()]
    weights = transform.sampling_weights[class_name]
    counts = transform.sampling_counts[class_name]
    uncertainty = uncertainty.detach().cpu()
    weights[bool_masked_pos[0]] = (weights[bool_masked_pos[0]] * counts[bool_masked_pos[0]] + uncertainty) / (
                counts[bool_masked_pos[0]] + 1)
    counts[bool_masked_pos[0]] += 1
    transform.sampling_weights[class_name] = weights
    transform.sampling_counts[class_name] = counts
    return transform


def train_one_epoch(model: torch.nn.Module, tokenizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, is_multiroad=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if epoch >= 100:  # dynamic masking
        if not is_multiroad:
            data_loader.dataset.transform.sampling_by_weights = True
        else:
            data_loader.dataset.transform16.sampling_by_weights = True
            data_loader.dataset.transform32.sampling_by_weights = True

    for step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        if is_multiroad:
            batch16, batch32, target = batch_data
        else:
            batch, target = batch_data

        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if is_multiroad:
            tokenizer16, tokenizer32 = tokenizer
            images16, labels16, bool_masked_pos16, mask_weights16 = get_img_and_label(epoch, tokenizer16, batch16, device)
            images32, labels32, bool_masked_pos32, mask_weights32 = get_img_and_label(epoch, tokenizer32, batch32, device)
        else:
            images, labels, bool_masked_pos, mask_weights = get_img_and_label(epoch, tokenizer, batch, device)

        with torch.cuda.amp.autocast():
            # outputs: (mask_N, 8192); labels: (mask_N)
            if not is_multiroad:
                outputs, x_decoder = model(images, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
                uncertainty = nn.CrossEntropyLoss(reduction='none')(input=outputs, target=labels)
                if epoch >= 100:
                    loss = uncertainty * mask_weights
                else:
                    loss = uncertainty
                loss = torch.mean(loss)

                # updating sampling weights
                data_loader.dataset.transform = updat_sampling_weights(data_loader.dataset.transform, target, uncertainty, bool_masked_pos)
            else:
                outputs16, outputs32, x_decoder16, x_decoder32 = model(images16, images32, bool_masked_pos16, bool_masked_pos32, return_all_tokens=False)

                # ce loss (reconstruction)
                uncertainty16 = nn.CrossEntropyLoss(reduction='none')(input=outputs16, target=labels16)
                uncertainty32 = nn.CrossEntropyLoss(reduction='none')(input=outputs32, target=labels32)
                if epoch >= 100:
                    loss16 = uncertainty16 * mask_weights16
                    loss32 = uncertainty32 * mask_weights32
                else:
                    loss16, loss32 = uncertainty16, uncertainty32
                loss_cs = torch.mean(loss16) + torch.mean(loss32)

                # feature sim loss
                bs = images16.size(0)
                t = torch.ones(bs).to(device)
                feature_sim = nn.CosineEmbeddingLoss(reduction='none')(x_decoder16, x_decoder32, t)

                # kl loss
                num_classes16, num_classes32 = outputs16.size(1), outputs32.size(1)
                labels16_onehot, labels32_onehot = F.one_hot(labels16, num_classes=num_classes16), F.one_hot(labels32, num_classes=num_classes32)
                kl_loss_16 = nn.BCEWithLogitsLoss()(outputs16, labels16_onehot.to(torch.float32))
                kl_loss_32 = nn.BCEWithLogitsLoss()(outputs32, labels32_onehot.to(torch.float32))
                kl_loss =  kl_loss_16 + kl_loss_32

                loss = loss_cs + feature_sim + kl_loss

                # updating sampling weights
                data_loader.dataset.transform16 = updat_sampling_weights(data_loader.dataset.transform16, target, uncertainty16, bool_masked_pos16)
                data_loader.dataset.transform32 = updat_sampling_weights(data_loader.dataset.transform32, target, uncertainty32, bool_masked_pos32)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            #sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if not is_multiroad:
            mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
        else:
            mlm_acc = (outputs16.max(-1)[1] == labels16).float().mean().item()

        metric_logger.update(mlm_acc=mlm_acc)
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
