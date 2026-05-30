from cellpose.cli import get_arg_parser as get_cellpose_arg_parser


MODEL_SAVE_ROOT_FLAG = "--model_save_root"


def _has_option(parser, option):
    return any(option in action.option_strings for action in parser._actions)


def add_astra_arguments(parser):
    if not _has_option(parser, MODEL_SAVE_ROOT_FLAG):
        parser.add_argument(
            MODEL_SAVE_ROOT_FLAG,
            default=[],
            type=str,
            help=(
                "ASTRA training checkpoint parent. When provided, model "
                "checkpoints are written under this directory instead of the "
                "training image directory."
            ),
        )
    return parser


def get_arg_parser():
    return add_astra_arguments(get_cellpose_arg_parser())
