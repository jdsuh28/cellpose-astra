import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cellpose.astra import __main__ as astra_main
from cellpose.astra.cli import get_arg_parser as get_astra_arg_parser
from cellpose.cli import get_arg_parser as get_cellpose_arg_parser


class _DummyModel:
    def __init__(self, *args, **kwargs):
        self.net = object()
        self.pretrained_model = None


class TrainingSaveRootTest(unittest.TestCase):

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
            model_save_root=[],
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
            checkpoint = (
                Path(kwargs["save_path"])
                / "models"
                / f"{kwargs['model_name']}_epoch_{kwargs['n_epochs']:04d}"
            )
            return checkpoint, [0.0], [0.0]

        logger = argparse.Namespace(critical=lambda *_args, **_kwargs: None)
        with mock.patch.object(astra_main.io, "load_train_test_data", fake_load_train_test_data), \
                mock.patch.object(astra_main.models, "CellposeModel", _DummyModel), \
                mock.patch.object(astra_main.astra_train, "train_seg", fake_train_seg):
            model = astra_main._train_cellposemodel_cli(
                args,
                logger=logger,
                image_filter=[],
                device=None,
                pretrained_model="cpsam",
                normalize=True,
            )
        return captured, model

    def test_model_save_root_cli_argument_is_astra_only(self):
        args = get_astra_arg_parser().parse_args(["--train", "--dir", "/tmp/astra-train"])

        self.assertEqual([], args.model_save_root)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            get_cellpose_arg_parser().parse_args([
                "--train",
                "--dir",
                "/tmp/astra-train",
                "--model_save_root",
                "/tmp/astra-output",
            ])

    def test_training_save_path_defaults_to_dir_models_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            train_dir.mkdir()
            args = self._minimal_train_args(train_dir)

            captured, model = self._capture_training_save_path(args)

            self.assertEqual(str(train_dir.resolve()), captured["save_path"])
            self.assertEqual(
                train_dir.resolve() / "models" / "astra_test_model_epoch_0001",
                Path(model.pretrained_model),
            )

    def test_model_save_root_writes_sibling_models_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "training" / "nucleus"
            train_dir = training_root / "train"
            test_dir = training_root / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir()
            args = self._minimal_train_args(
                train_dir,
                test_dir=str(test_dir),
                model_save_root=str(training_root),
            )

            captured, model = self._capture_training_save_path(args)

            self.assertEqual(str(training_root.resolve()), captured["save_path"])
            self.assertEqual(
                training_root.resolve() / "models" / "astra_test_model_epoch_0001",
                Path(model.pretrained_model),
            )
            self.assertNotIn("train/models", str(model.pretrained_model))

    def test_upstream_cellpose_runtime_files_do_not_contain_astra_hooks(self):
        root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "cellpose/train.py",
            "cellpose/cli.py",
            "cellpose/__main__.py",
        ):
            text = (root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("ASTRA START", text)
            self.assertNotIn("model_save_root", text)
            self.assertNotIn("cellpose.astra", text)


if __name__ == "__main__":
    unittest.main()
