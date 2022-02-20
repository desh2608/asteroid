import torch
from torch.utils import data
import json

from lhotse.cut import MonoCut, MixedCut, CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.utils import compute_num_samples

DATASET = "LhotseCSS"


class LhotseCSSDataset(data.Dataset):
    """Dataset class for continuous source separation training based on Lhotse manifests.

    Args:
        mix_jsonl (str): The path to JSONL file containing mixed sessions.
        src_jsonl (str): The path to JSONL file containing sources. Each entry must
            have the keys "cuts", "start", and "end".
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
    """

    dataset_name = "LhotseCSS"

    def __init__(self, mix_jsonl, src_jsonl, n_src=2, sample_rate=8000):
        super(LhotseCSSDataset, self).__init__()
        # Task setting
        self.mix_jsonl = mix_jsonl
        self.src_jsonl = src_jsonl
        self.sample_rate = sample_rate
        self.n_src = n_src

        mix_info = []
        src_info = {}
        with open(mix_jsonl, "r") as f_mix, open(src_jsonl, "r") as f_src:
            for line_mix, line_src in zip(f_mix, f_src):
                data_mix = json.loads(line_mix)
                # Sometimes the mixture may have just 1 cut
                if data_mix["type"] == "MonoCut":
                    mix_info.append(MonoCut.from_dict(data_mix))
                else:
                    mix_info.append(MixedCut.from_dict(data_mix))
                data_src = json.loads(line_src)
                data_src["cuts"] = [MonoCut.from_dict(c) for c in data_src["cuts"]]
                src_info[data_mix["id"]] = data_src

        self.mix = CutSet.from_cuts(mix_info)
        self.sources = src_info

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, cuts):
        """Gets a mixture/sources pair.
        Returns:
            mixture, [source_signals], [segment_boundaries]
            See: https://github.com/fgnt/graph_pit
        """
        # Load mixture
        features, feature_lens = collate_audio(cuts)
        # Load sources and boundaries
        sources = [
            [
                torch.from_numpy(s.load_audio()).squeeze()
                for s in self.sources[idx]["cuts"]
            ]
            for idx in cuts.ids
        ]
        boundaries = [
            [
                (
                    compute_num_samples(start, self.sample_rate),
                    compute_num_samples(end, self.sample_rate),
                )
                for start, end in zip(
                    self.sources[idx]["start"], self.sources[idx]["end"]
                )
            ]
            for idx in cuts.ids
        ]
        targets = []
        for s, b, l in zip(sources, boundaries, feature_lens.tolist()):
            # ensure that sources and boundaries are sorted in order of start times
            sources_and_boundaries = sorted(zip(s, b), key=lambda x: x[1][0])
            s, b = zip(*sources_and_boundaries)
            targets.append(
                {
                    "sources": s,
                    "boundaries": b,
                    "length": l,
                }
            )
        return features, targets

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "sep_continuous"
        infos["licences"] = None
        return infos
