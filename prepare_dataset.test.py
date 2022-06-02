import argparse

from museparation.util.create_hdf import create_hdf
from museparation.scripts.get_musdb import get_musdb, get_musdbhq


def main(args):

    # Get musdb dataset
    if args.HQ:
        musdb = get_musdbhq(args.musdb_path)
    else:
        throw("not yet implemented")

    # Create hdf
    if args.shuffle:
        create_hdf(args.hdf_dir, "train", musdb["train"], source=None)
    else:
        create_hdf(args.hdf_dir, "train", musdb["train"], source="mixture")
    create_hdf(args.hdf_dir, "val", musdb["val"], source="mixture")
    create_hdf(args.hdf_dir, "test", musdb["test"], source="mixture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--musdb_path", type=str, required=True, help="Path to musdb18 dataset"
    )
    parser.add_argument("--hdf_dir", type=str, default="hdf", help="Path to hdf dir")
    parser.add_argument(
        "--HQ",
        action="store_true",
        default=False,
        help="check to use musdb18hq instead of standard musdb18 dataset",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=["bass", "drums", "other", "vocals"],
        help='The list of instruments to separate (default = "bass drums other vocals")',
    )
    parser.add_argument(
        "--mixture", type=str, default="mixture", help="The name for mixture file"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="switch for shuffling the dataset, if set then --mixture will be ignored",
    )

    args = parser.parse_args()

    main(args)
