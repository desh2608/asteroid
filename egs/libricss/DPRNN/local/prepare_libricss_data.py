#!/usr/local/bin/python3
from pathlib import Path
import logging
import json

from lhotse.recipes.libricss import prepare_libricss
from lhotse import CutSet

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def read_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create LibriCSS manifests (clean and replayed)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing LibriCSS (for_release) corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write meeting manifests to",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Generate replayed mixtures
    logging.info("Generating replayed mixtures")
    manifests = prepare_libricss(args.corpus_dir, type="mdm")
    cuts_replayed = CutSet.from_manifests(recordings=manifests["recordings"])
    cuts_replayed.to_file(args.output_dir / "libricss_replayed.jsonl")

    # Generate clean mixtures
    logging.info("Generating clean mixtures")
    manifests = prepare_libricss(args.corpus_dir, type="ihm-mix")
    cuts_clean = CutSet.from_manifests(recordings=manifests["recordings"])
    cuts_clean.to_file(args.output_dir / "libricss_clean.jsonl")

    # Generate sources
    logging.info("Generating sources")
    manifests = prepare_libricss(args.corpus_dir, type="ihm")
    cuts_sources = CutSet.from_manifests(
        recordings=manifests["recordings"], supervisions=manifests["supervisions"]
    )
    with open(args.output_dir / "libricss_sources.jsonl", "w") as f:
        for cut in cuts_sources:
            cuts = CutSet.from_cuts(
                cut.trim_to_supervisions(keep_overlapping=False)
            ).drop_supervisions()
            start = [c.start for c in cuts]
            end = [c.end for c in cuts]
            data = {
                "cuts": [cut.to_dict() for cut in cuts],
                "start": start,
                "end": end,
            }
            f.write(json.dumps(data) + "\n")
