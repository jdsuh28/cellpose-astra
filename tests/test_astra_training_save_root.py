import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cellpose import __main__ as cellpose_main
from cellpose.cli import get_arg_parser


class _DummyModel:
    def __init__(self, *args, **kwargs):
        self.net = object()
        self.pretrained_model = None


class AstraTrainingSaveRootTest(unittest.TestCase):

    def _minimal_train_args(self, train_dir, **overrides):
        values = dict(
            test_dir=[],
            file_list=[],
            mask_filter="_masks",
            look_one_level_down=False,
            channel_axis=None,
            learning_rate=1e-5,
            weight_decay=0.1,
            SGD=0,
            n_epochs=1,
            train_batch_size=1,
            min_train_masks=1,
            nimg_per_epoch=None,
            nimg_test_per_epoch=None,
            save_every=1,
            save_each=False,
            model_name_out="astra_test_model",
            astra_model_save_root=[],
            dir=str(train_dir),
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def _capture_training_save_path(self, args):
        captured = {}

        def fake_load_train_test_data(*_args, **_kwargs):
            return ["image"], ["label"], ["train_file"], None, None, None

        def fake_train_seg(*_args, **kwargs):
            captured.update(kwargs)
            return str(Path(kwargs["save_path"]) / "models" / kwargs["model_name"]), [0.0], [0.0]

        logger = argparse.Namespace(critical=lambda *_args, **_kwargs: None)
        with mock.patch.object(cellpose_main.io, "load_train_test_data", fake_load_train_test_data), \
                mock.patch.object(cellpose_main.models, "CellposeModel", _DummyModel), \
                mock.patch.object(cellpose_main.train, "train_seg", fake_train_seg):
            model = cellpose_main._train_cellposemodel_cli(
                args,
                logger=logger,
                image_filter=[],
                device=None,
                pretrained_model="cpsam",
                normalize=True,
            )
        return captured, model

    def test_astra_model_save_root_cli_argument_is_optional_and_defaults_to_upstream_behavior(self):
        args = get_arg_parser().parse_args(["--train", "--dir", "/tmp/astra-train"])

        self.assertEqual([], args.astra_model_save_root)

    def test_training_save_path_defaults_to_dir_models_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            train_dir.mkdir()
            args = self._minimal_train_args(train_dir)

            captured, model = self._capture_training_save_path(args)

            self.assertEqual(str(train_dir.resolve()), captured["save_path"])
            self.assertEqual(
                str(train_dir.resolve() / "models" / "astra_test_model"),
                model.pretrained_model,
            )

    def test_astra_training_save_root_writes_sibling_models_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "training" / "nucleus"
            train_dir = training_root / "train"
            test_dir = training_root / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir()
            args = self._minimal_train_args(
                train_dir,
                test_dir=str(test_dir),
                astra_model_save_root=str(training_root),
            )

            captured, model = self._capture_training_save_path(args)

            self.assertEqual(str(training_root.resolve()), captured["save_path"])
            self.assertEqual(
                str(training_root.resolve() / "models" / "astra_test_model"),
                model.pretrained_model,
            )
            self.assertNotIn("train/models", model.pretrained_model)


if __name__ == "__main__":
    unittest.main()
