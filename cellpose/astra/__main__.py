import logging
import os

import numpy as np

from cellpose import io, models
from cellpose.__main__ import main as cellpose_main
from cellpose.astra import train as astra_train
from cellpose.astra.cli import get_arg_parser
from cellpose.version import version_str


def main():
    args = get_arg_parser().parse_args()

    if args.version:
        print(version_str)
        return

    if not args.train:
        cellpose_main()
        return

    if args.verbose:
        from cellpose.io import logger_setup
        logger, _log_file = logger_setup()
    else:
        print(">>>> !LOGGING OFF BY DEFAULT! To see cellpose progress, set --verbose")
        print("No --verbose => no progress or info printed")
        logger = logging.getLogger(__name__)

    image_filter = args.img_filter if len(args.img_filter) > 0 else None

    device, _gpu = models.assign_device(use_torch=True, gpu=args.use_gpu,
                                        device=args.gpu_device)

    if (args.pretrained_model is None or args.pretrained_model == "None" or
            args.pretrained_model == "False" or args.pretrained_model == "0"):
        pretrained_model = "cpsam"
        logger.warning("training from scratch is disabled, using 'cpsam' model")
    else:
        pretrained_model = args.pretrained_model

    if args.pretrained_model_ortho:
        logger.warning(
            "the '--pretrained_model_ortho' flag is deprecated in v4.0.1+ and no longer used")
    if args.train_size:
        logger.warning("the '--train_size' flag is deprecated in v4.0.1+ and no longer used")
    if args.chan or args.chan2:
        logger.warning("--chan and --chan2 are deprecated, all channels are used by default")
    if args.all_channels:
        logger.warning("the '--all_channels' flag is deprecated in v4.0.1+ and no longer used")
    if args.restore_type:
        logger.warning("the '--restore_type' flag is deprecated in v4.0.1+ and no longer used")
    if args.transformer:
        logger.warning("the '--tranformer' flag is deprecated in v4.0.1+ and no longer used")
    if args.invert:
        logger.warning("the '--invert' flag is deprecated in v4.0.1+ and no longer used")
    if args.chan2_restore:
        logger.warning("the '--chan2_restore' flag is deprecated in v4.0.1+ and no longer used")
    if args.diam_mean:
        logger.warning("the '--diam_mean' flag is deprecated in v4.0.1+ and no longer used")
    if args.train_size:
        logger.warning("the '--train_size' flag is deprecated in v4.0.1+ and no longer used")

    if args.norm_percentile is not None:
        value1, value2 = args.norm_percentile
        normalize = {"percentile": (float(value1), float(value2))}
    else:
        normalize = (not args.no_norm)

    if args.save_each and not args.save_every:
        raise ValueError("ERROR: --save_each requires --save_every")

    if len(args.image_path) > 0 and args.train:
        raise ValueError("ERROR: cannot train model with single image input")

    _train_cellposemodel_cli(args, logger, image_filter, device, pretrained_model, normalize)


def _model_save_path(args):
    model_save_root = getattr(args, "model_save_root", [])
    if isinstance(model_save_root, str) and len(model_save_root) > 0:
        return os.path.realpath(model_save_root)
    return os.path.realpath(args.dir)


def _train_cellposemodel_cli(args, logger, image_filter, device, pretrained_model, normalize):
    test_dir = None if len(args.test_dir) == 0 else args.test_dir
    images, labels, image_names, train_probs = None, None, None, None
    test_images, test_labels, image_names_test, test_probs = None, None, None, None
    compute_flows = False
    if len(args.file_list) > 0:
        if os.path.exists(args.file_list):
            dat = np.load(args.file_list, allow_pickle=True).item()
            image_names = dat["train_files"]
            image_names_test = dat.get("test_files", None)
            train_probs = dat.get("train_probs", None)
            test_probs = dat.get("test_probs", None)
            compute_flows = dat.get("compute_flows", False)
            load_files = False
        else:
            logger.critical(f"ERROR: {args.file_list} does not exist")
    else:
        output = io.load_train_test_data(args.dir, test_dir, image_filter,
                                         args.mask_filter,
                                         args.look_one_level_down)
        images, labels, image_names, test_images, test_labels, image_names_test = output
        load_files = True

    model = models.CellposeModel(device=device, pretrained_model=pretrained_model)

    cpmodel_path = astra_train.train_seg(
        model.net, images, labels, train_files=image_names,
        test_data=test_images, test_labels=test_labels,
        test_files=image_names_test, train_probs=train_probs,
        test_probs=test_probs, compute_flows=compute_flows,
        load_files=load_files, normalize=normalize,
        channel_axis=args.channel_axis,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        SGD=args.SGD, n_epochs=args.n_epochs, batch_size=args.train_batch_size,
        min_train_masks=args.min_train_masks,
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
        save_path=_model_save_path(args),
        save_every=args.save_every,
        save_each=args.save_each,
        model_name=args.model_name_out)[0]
    model.pretrained_model = cpmodel_path
    return model


if __name__ == "__main__":
    main()
