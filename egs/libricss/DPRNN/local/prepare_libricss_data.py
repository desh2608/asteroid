#!/usr/local/bin/python3
from pathlib import Path
import logging
import json
from itertools import chain, groupby

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
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Generate replayed mixtures
    logging.info("Generating replayed mixtures")
    manifests = prepare_libricss(args.corpus_dir, type="mdm")
    cuts_replayed = CutSet.from_manifests(recordings=manifests["recordings"])
    # Mix all cuts from the same recording but from different channels
    cuts_replayed = cuts_replayed.mix_same_recording_channels()
    # Reassign the id (since they got randomized)
    cuts_replayed = CutSet.from_cuts(
        cut.with_id(cut.tracks[0].cut.recording_id) for cut in cuts_replayed
    )
    cuts_replayed.to_file(args.output_dir / "libricss_replayed.jsonl")

    # Generate clean mixtures
    logging.info("Generating clean mixtures")
    manifests = prepare_libricss(args.corpus_dir, type="ihm-mix")
    cuts_clean = CutSet.from_manifests(recordings=manifests["recordings"])
    # Modify ids: '0L_session1-0-0' -> '0L_session1'
    cuts_clean = cuts_clean.modify_ids(lambda cut_id: cut_id.split("-")[0])
    # Sort the cuts in the same order as the replayed mixtures
    cuts_clean = cuts_clean.sort_like(cuts_replayed)
    cuts_clean.to_file(args.output_dir / "libricss_clean.jsonl")

    # Generate sources
    logging.info("Generating sources")
    manifests = prepare_libricss(args.corpus_dir, type="ihm")
    cuts_sources = CutSet.from_manifests(
        recordings=manifests["recordings"], supervisions=manifests["supervisions"]
    )
    # Each cut contains supervisions from 1 speaker (total 60 x 8 = 480 cuts)
    cuts_sources = list(
        chain.from_iterable(
            [cut.trim_to_supervisions(keep_overlapping=False) for cut in cuts_sources]
        )
    )
    # Group the cuts by recording id
    cuts_sources = {
        r: list(g)
        for r, g in groupby(
            sorted(cuts_sources, key=lambda cut: cut.recording_id),
            key=lambda cut: cut.recording_id,
        )
    }
    with open(args.output_dir / "libricss_sources.jsonl", "w") as f:
        # We iterate over all ids in cuts_replayed so that the list is sorted in the
        # same order as the replayed mixtures
        for id in cuts_replayed.ids:
            cuts = CutSet.from_cuts(cuts_sources[id]).drop_supervisions()
            start = [c.start for c in cuts]
            end = [c.end for c in cuts]
            data = {
                "cuts": [cut.to_dict() for cut in cuts],
                "start": start,
                "end": end,
            }
            f.write(json.dumps(data) + "\n")
