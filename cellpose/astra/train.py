import logging
import time
from pathlib import Path

import numpy as np
import torch

from cellpose import models
from cellpose.train import (
    _get_batch,
    _loss_fn_class,
    _loss_fn_seg,
    _process_train_test,
)
from cellpose.transforms import random_rotate_and_resize


train_logger = logging.getLogger(__name__)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=1e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None):
    """
    Train Cellpose segmentation with ASTRA checkpoint naming.

    The implementation tracks upstream Cellpose training behavior while writing
    only suffixed epoch checkpoints so ASTRA promotion can select from explicit
    training outputs.
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    original_net_dtype = net.dtype
    if net.dtype == torch.bfloat16:
        train_logger.info(">>> converting bfloat16 network to float32 for training")
        net.dtype = torch.float32

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files,
                              train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files,
                              test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files,
                              min_train_masks=min_train_masks,
                              compute_flows=compute_flows,
                              channel_axis=channel_axis,
                              normalize_params=normalize_params,
                              device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for _i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for _i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving ASTRA model checkpoints under {filename.parent}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        epoch_number = iepoch + 1
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files,
                                    labels_files=train_labels_files,
                                    **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                 scale_range=scale_range,
                                                 xy=(bsize, bsize))[:2]
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)

            with torch.autocast(device_type=device.type, dtype=net.dtype):
                y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            if y.shape[1] > 3:
                loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                loss += loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss *= len(imgi)

            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
        train_losses[iepoch] /= nimg_per_epoch

        lavgt = 0.
        if test_data is not None or test_files is not None:
            np.random.seed(42)
            if nimg_test != nimg_test_per_epoch:
                rperm = np.random.choice(np.arange(0, nimg_test),
                                         size=(nimg_test_per_epoch,), p=test_probs)
            else:
                rperm = np.random.permutation(np.arange(0, nimg_test))
            for ibatch in range(0, len(rperm), batch_size):
                with torch.no_grad():
                    net.eval()
                    inds = rperm[ibatch:ibatch + batch_size]
                    imgs, lbls = _get_batch(inds, data=test_data,
                                            labels=test_labels, files=test_files,
                                            labels_files=test_labels_files,
                                            **kwargs)
                    diams = np.array([diam_test[i] for i in inds])
                    rsc = diams / net.diam_mean.item() if rescale else np.ones(
                        len(diams), "float32")
                    imgi, lbl = random_rotate_and_resize(
                        imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                        xy=(bsize, bsize))[:2]
                    X = torch.from_numpy(imgi).to(device)
                    lbl = torch.from_numpy(lbl).to(device)

                    with torch.autocast(device_type=device.type, dtype=net.dtype):
                        y = net(X)[0]
                    loss = _loss_fn_seg(lbl, y, device)
                    if y.shape[1] > 3:
                        loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                        loss += loss3
                    test_loss = loss.item()
                    test_loss *= len(imgi)
                    lavgt += test_loss
            lavgt /= len(rperm)
            test_losses[iepoch] = lavgt
        lavg /= nsum
        train_logger.info(
            f"{epoch_number}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
        )
        lavg, nsum = 0, 0

        if save_every and epoch_number % save_every == 0 and epoch_number != n_epochs:
            filename0 = str(filename) + f"_epoch_{epoch_number:04d}"
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    final_checkpoint = str(filename) + f"_epoch_{n_epochs:04d}"
    train_logger.info(f"saving network parameters to {final_checkpoint}")
    net.save_model(final_checkpoint)
    if original_net_dtype != torch.float32:
        train_logger.info(
            f">>> converting network back to {original_net_dtype} after training"
        )
        net.dtype = original_net_dtype

    return Path(final_checkpoint), train_losses, test_losses
