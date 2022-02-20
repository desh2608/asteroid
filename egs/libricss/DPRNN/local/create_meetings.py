#!/usr/local/bin/python3
from pathlib import Path
from itertools import groupby
from collections import Counter
from tqdm import tqdm
import logging
import random
import json

import numpy as np
from scipy.stats import bernoulli

from lhotse.recipes.librispeech import prepare_librispeech
from lhotse import RecordingSet, SupervisionSet, CutSet, load_manifest
from lhotse.utils import fastcopy

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def read_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create meetings from LibriSpeech utterances"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing LibriSpeech corpus",
    )
    parser.add_argument(
        "--ctm-dir",
        type=Path,
        required=True,
        help="Directory containing CTM files (we use these to get strict utterance boundaries)",
    )
    parser.add_argument(
        "--rir-dir",
        type=Path,
        required=True,
        help="Directory containing RIR files (to simulate far-field audio)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Minimum number of speakers per meeting",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=3,
        help="Maximum number of speakers per meeting",
    )
    parser.add_argument(
        "--min-utts-per-session",
        type=int,
        default=3,
        help="Minimum number of utterances per session",
    )
    parser.add_argument(
        "--max-utts-per-session",
        type=int,
        default=5,
        help="Maximum number of utterances per session",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.3,
        help="Minimum overlap ratio in a meeting",
    )
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.5,
        help="Maximum overlap ratio in a meeting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write meeting manifests to",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def split_supervision_segment(segment, max_pause=0.1):
    """
    Split a supervision segment based on word alignments, i.e., break segments when the gap
    between them is longer than max_pause.
    """
    segments = []
    alignment_items = sorted(segment.alignment["word"], key=lambda x: x.start)
    start = alignment_items[0].start
    duration = alignment_items[0].duration
    text = alignment_items[0].symbol
    for item in alignment_items[1:]:
        if item.start <= start + duration + max_pause:
            text += f" {item.symbol}"
            duration += item.duration
        else:
            if len(text) > 0:
                segments.append(
                    fastcopy(
                        segment,
                        id=f"{segment.id}_{len(segments)}",
                        start=start,
                        duration=duration,
                        text=text,
                        alignment=None,
                    )
                )
            start = item.start
            duration = item.duration
            text = item.symbol
    if len(text) > 0:
        segments.append(
            fastcopy(
                segment,
                id=f"{segment.id}_{len(segments)}",
                start=start,
                duration=duration,
                text=text,
                alignment=None,
            )
        )
    return segments


def read_single_utterances(args, part):
    if (args.output_dir / f"orig_cuts_{part}.jsonl").exists():
        manifests = load_manifest(args.output_dir / f"orig_cuts_{part}.jsonl")
        return manifests

    # Get Lhotse manifests for LibriSpeech
    manifests = prepare_librispeech(
        args.corpus_dir,
        dataset_parts=part,
        num_jobs=4,
        output_dir=args.output_dir,
    )

    # Add supervision alignments using CTM files
    manifests[part]["supervisions"] = manifests[part][
        "supervisions"
    ].with_alignment_from_ctm(args.ctm_dir / f"{part.replace('-', '_')}.ctm")

    # Create strict supervisions using the alignments
    segments = []
    for seg in manifests[part]["supervisions"]:
        segments += split_supervision_segment(seg)
    manifests[part]["supervisions"] = SupervisionSet.from_segments(segments)

    # Create cuts from the recordings and supervisions
    cuts = CutSet.from_manifests(
        recordings=manifests[part]["recordings"],
        supervisions=manifests[part]["supervisions"],
    ).trim_to_supervisions(keep_overlapping=False)
    cuts.to_file(args.output_dir / f"orig_cuts_{part}.jsonl")
    return cuts


def careful_shuffle(cuts):
    """
    We use the following algorithm to arrange utterances so that adjacent utterances have
    different speakers as far as possible:
    https://stackoverflow.com/questions/30881187/algorithm-to-shuffle-a-list-to-minimise-equal-neighbours
    """
    if len(cuts) < 2:
        return cuts

    speakers = cuts.speakers
    cuts = [cuts[id] for id in cuts.ids]
    c = {
        spk: list(filter(lambda cut: cut.supervisions[0].speaker == spk, cuts))
        for spk in speakers
    }

    output = []
    last = None
    while True:
        # All we need are the current 2 most commonly occurring items. This
        # works even if there's only 1 or even 0 items left, because the
        # Counter object will still return the requested number of results,
        # with count == 0.
        common_items = Counter(c.keys()).most_common(2)
        avail_items = [key for key, count in common_items if count]

        if not avail_items:
            # No more items to process.
            break

        # Just reverse the list if we just saw the first item. This simplies
        # the logic in case we actually only have 1 type of item left (in which
        # case we have no choice but to choose it).
        if avail_items[0] == last:
            avail_items.reverse()

        # We've found the next item. Add it to the output and update the
        # counter.
        next_spk = avail_items[0]
        next_cut = c[next_spk].pop()
        if len(c[next_spk]) == 0:
            del c[next_spk]
        output.append(next_cut)
        last = next_spk

    return output


def give_timing(cuts, overlap_time_ratio=0.3, sil_prob=0.2, sil_dur=[0.3, 2.0]):
    # Reorder cuts to avoid adjacent cuts spoken by the same speaker
    cuts_reorg = careful_shuffle(cuts)

    # Calculate the total length and derive the overlap time budget.
    total_len = np.sum(np.array([cut.duration for cut in cuts_reorg]))
    total_overlap_time = total_len * overlap_time_ratio / (1 + overlap_time_ratio)

    # Determine where to do overlap.
    nutts = len(cuts)
    to_overlap = bernoulli.rvs(1 - sil_prob, size=nutts - 1).astype(bool).tolist()
    noverlaps = sum(to_overlap)

    # Distribute the budget to each utterance boundary with the "stick breaking" approach.
    probs = []
    rem = 1
    for i in range(noverlaps - 1):
        p = random.betavariate(1, 5)
        probs.append(rem * p)
        rem *= 1 - p
    probs.append(rem)
    random.shuffle(probs)

    idx = -1
    overlap_times = [0.0]
    for b in to_overlap:
        if b:
            idx += 1
            overlap_times.append(probs[idx] * total_overlap_time)
        else:
            overlap_times.append(-np.random.uniform(low=sil_dur[0], high=sil_dur[1]))

    # Get all speakers.
    speakers = cuts.speakers

    # Determine the offset values while ensuring that there is no overlap between multiple
    # utterances spoken by the same person.
    new_cuts = []
    new_offsets = []
    offset = 0
    last_utt_end = {spkr: 0.0 for spkr in speakers}
    last_utt_end_times = sorted(
        list(last_utt_end.values()), reverse=True
    )  # all zero (of course!)
    actual_overlap_time = 0
    for cut, ot in zip(cuts_reorg, overlap_times):
        spkr = cut.supervisions[0].speaker

        if len(last_utt_end_times) > 1:
            # second term for ensuring same speaker's utterances do not overlap.
            # third term for ensuring the maximum number of overlaps is two.
            ot = min(ot, offset - last_utt_end[spkr], offset - last_utt_end_times[1])

        offset -= ot
        if offset < 0:
            raise ValueError("Negative offset occurred")
        actual_overlap_time += max(ot, 0)

        new_cuts.append(cut)
        new_offsets.append(offset)

        offset += cut.duration
        last_utt_end[spkr] = offset

        last_utt_end_times = sorted(list(last_utt_end.values()), reverse=True)
        offset = last_utt_end_times[0]

    actual_overlap_time_ratio = actual_overlap_time / (total_len - actual_overlap_time)

    return new_cuts, new_offsets, actual_overlap_time_ratio


def generate_meetings(args, cuts):
    meetings = []
    # Group cuts by speaker
    speakers = cuts.speakers
    cuts_by_speaker = {
        spk: cuts.filter(lambda cut: cut.supervisions[0].speaker == spk)
        for spk in speakers
    }
    with tqdm(total=len(cuts)) as pbar:
        while len(cuts_by_speaker) > 0:
            # Select total number of utterances (N)
            N = random.randint(args.min_utts_per_session, args.max_utts_per_session)
            # Select total number of speakers (M)
            M = random.randint(args.min_speakers, args.max_speakers)
            # Select M speakers from remaining speakers
            remaining_speakers = list(cuts_by_speaker.keys())
            selected_spk = random.sample(
                remaining_speakers, min(M, len(remaining_speakers))
            )
            # Select utterances for each speaker
            utts = []
            for i, spk in enumerate(selected_spk):
                # For first N%M speakers, we choose N//M + 1 utterances. For others, we choose N//M.
                n = N // M + int(i < N % M)
                speaker_cuts = cuts_by_speaker[spk].sample(n)
                if isinstance(speaker_cuts, CutSet):
                    utts += list(speaker_cuts.ids)
                else:
                    utts.append(speaker_cuts.id)

            selected_utts = cuts.subset(cut_ids=utts)

            # Get timings for desired overlap ratio
            ovl_ratio = random.uniform(args.min_overlap_ratio, args.max_overlap_ratio)
            try:
                new_cuts, offsets, actual_ovl_ratio = give_timing(
                    selected_utts, overlap_time_ratio=ovl_ratio
                )
            except ValueError:
                logging.warn(
                    "Negative offset occurred while generating meeting. Trying again!"
                )
                continue

            # Remove selected utterances from remaining speakers
            for spk in selected_spk:
                # Remove selected cuts from the list of speaker cuts
                cuts_by_speaker[spk] = cuts_by_speaker[spk].filter(
                    lambda cut: cut.id not in utts
                )
                if len(cuts_by_speaker[spk]) == 0:
                    del cuts_by_speaker[spk]

            meetings.append(
                {
                    "cuts": new_cuts,
                    "offsets": offsets,
                    "ovl_ratio": actual_ovl_ratio,
                }
            )
            # Update progress bar
            pbar.update(len(new_cuts))
    return meetings


def mix_meeting_cuts(args, meetings):
    # Get the rirs (load from cached if available)
    rir_manifest_path = args.output_dir / "rir_manifest.jsonl"
    if rir_manifest_path.exists():
        logging.info(f"Loading cached RIRs. Delete {rir_manifest_path} to regenerate.")
        rirs = load_manifest(rir_manifest_path)
    else:
        rirs = RecordingSet.from_dir(args.rir_dir, "*.wav")
        rirs.to_file(rir_manifest_path)

    # RIR's are named as mix_0000001_mic-0_source-0.wav. For any meeting, we will randomly
    # select one of the RIRs, and then change the "source" part for each speaker.
    # Group the RIR's by id and mic
    rirs_by_id = {
        id: list(rirs)
        for id, rirs in groupby(
            sorted(rirs, key=lambda r: r.id),
            lambda r: r.id.rsplit("_", 1)[0],
        )
    }

    mixed_meetings_clean = []
    mixed_meetings_rvb = []
    for meeting in meetings:

        # Select a random RIR
        selected_rir_id = random.choice(list(rirs_by_id.keys()))
        selected_rirs = rirs_by_id[selected_rir_id]

        # Assign a random source to each speaker
        speakers = set([cut.supervisions[0].speaker for cut in meeting["cuts"]])
        spk_rirs = {spk: selected_rirs[i] for i, spk in enumerate(speakers)}

        # Mix the meeting cuts
        first_cut = meeting["cuts"][0]
        first_cut_rvb = first_cut.reverb_rir(
            spk_rirs[first_cut.supervisions[0].speaker]
        )
        mixed_cut_clean = first_cut
        mixed_cut_rvb = first_cut_rvb
        for cut, offset in zip(meeting["cuts"][1:], meeting["offsets"][1:]):
            cut_rvb = cut.reverb_rir(spk_rirs[cut.supervisions[0].speaker])
            mixed_cut_clean = mixed_cut_clean.mix(
                cut, offset_other_by=offset, allow_padding=True
            )
            mixed_cut_rvb = mixed_cut_rvb.mix(
                cut_rvb, offset_other_by=offset, allow_padding=True
            )
        mixed_meetings_clean.append(mixed_cut_clean)
        mixed_meetings_rvb.append(mixed_cut_rvb)

    return CutSet.from_cuts(mixed_meetings_clean), CutSet.from_cuts(mixed_meetings_rvb)


def write_meeting_sources(args, meetings, part):
    """
    This will be used to create the training targets for graph-PIT. Each JSONL entry
    will contain the following fields:
    - cuts: clean cuts present in the meeting
    - start: start offset of the clean cuts in the meeting
    - end: end offset of the clean cuts in the meeting
    - ovl_ratio: overlap ratio of the meeting
    """
    with open(args.output_dir / f"{part}_sources.jsonl", "w") as f:
        for meeting in meetings:
            cs = [cut.to_dict() for cut in meeting["cuts"]]
            start = meeting["offsets"]
            end = [offset + cut.duration for offset, cut in zip(start, meeting["cuts"])]
            ovl_ratio = meeting["ovl_ratio"]
            data = {
                "cuts": cs,
                "start": start,
                "end": end,
                "ovl_ratio": ovl_ratio,
            }
            f.write(json.dumps(data) + "\n")
    return


if __name__ == "__main__":
    args = read_args()

    assert (
        args.max_speakers <= 3 or args.rir_dir is None
    ), "Only 3 speakers supported since RIRs are only generated for 3 sources"

    # Set random seed
    random.seed(args.random_seed)

    # Generate train and dev meetings
    for part in ["train-clean-100", "dev-clean"]:
        single_cuts = read_single_utterances(args, part)
        logging.info(part)
        meetings = generate_meetings(args, single_cuts)
        logging.info(f"{len(meetings)} meetings generated")
        mixed_cuts_clean, mixed_cuts_reverb = mix_meeting_cuts(args, meetings)
        logging.info(
            f"Writing mixed cuts to {args.output_dir}/{part}_mixed_clean.jsonl "
            f"and {args.output_dir}/{part}_mixed_reverb.jsonl"
        )
        mixed_cuts_clean.to_file(args.output_dir / f"{part}_mixed_clean.jsonl")
        mixed_cuts_reverb.to_file(args.output_dir / f"{part}_mixed_reverb.jsonl")
        logging.info(
            f"Writing mixed cut sources to {args.output_dir}/{part}_sources.jsonl"
        )
        write_meeting_sources(args, meetings, part)
