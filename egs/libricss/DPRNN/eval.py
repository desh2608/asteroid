import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from asteroid import DPRNNTasNet
from asteroid.data.libricss_dataset import LibriCSSDataset
from asteroid.losses import GraphPITLossWrapper
from asteroid.utils import tensors_to_device
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from asteroid.dsp import beamforming
from asteroid_filterbanks import make_enc_dec
import asteroid_filterbanks.transforms as af_transforms

from graph_pit.loss.optimized import optimized_graph_pit_source_aggregated_sdr_loss
from lhotse.dataset.sampling import SimpleCutSampler
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["sep_clean", "sep_reverb"],
    help="Evaluate clean or replayed LibriCSS mixture",
)
parser.add_argument(
    "--test_dir",
    type=str,
    required=True,
    help="Test directory including the json files",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--decode_dir", default="exp/tmp/decode", help="Decode directory to save results"
)
parser.add_argument(
    "--n_save_ex",
    type=int,
    default=-1,
    help="Number of audio examples to save, -1 means all",
)
parser.add_argument(
    "--window-size",
    type=float,
    default=4.0,
    help="Window size in seconds",
)
parser.add_argument(
    "--hop-size",
    type=float,
    default=2.0,
    help="Window size in seconds",
)
parser.add_argument(
    "--multi-channel",
    action="store_true",
    help="If true, use multi-channel beamforming",
)


class TimeInvariantBeamWrapper(torch.nn.Module):
    def __init__(self, monaural_sep_model, fft_win=1024, hop_size=256, chunk_size=3):
        super(TimeInvariantBeamWrapper, self).__init__()
        self.monaural_sep = monaural_sep_model
        self.enc, self.dec = make_enc_dec(
            "stft",
            kernel_size=fft_win,
            n_filters=fft_win,
            stride=hop_size,
            window=torch.hamming_window(fft_win),
        )
        self.beamformer = beamforming.SoudenMVDRBeamformer()
        # chunk size in seconds for applying the beamformer
        self.chunk_size = chunk_size

    def get_mask(self, mixture, estimate):
        # Return WLM mask
        assert mixture.dtype in [torch.complex32, torch.complex64, torch.complex128]
        assert estimate.dtype in [torch.complex32, torch.complex64, torch.complex128]
        mask = (estimate.abs() ** 2) / (mixture.abs() ** 2 + 1e-8)
        return mask

    def forward(self, mixture, ref_channel=0):
        bsz, mics, samples = mixture.shape
        # apply separator only on ref channel
        estimates = self.monaural_sep(mixture[:, ref_channel].unsqueeze(1))  # B x 2 x T

        # Move tensors to CPU for beamforming (torch.einsum gives OOM sometimes)
        estimates = estimates.to(torch.device("cpu"))
        mixture = mixture.to(torch.device("cpu"))

        # Compute mixture obtained by summing the estimates
        mixture_from_estimates = estimates.sum(1, keepdim=True)  # B x 1 x T

        # get STFTs for mixture and estimates
        mixture_stft = af_transforms.to_torch_complex(
            self.enc(mixture)
        )  # B x C x F x N
        mixture_from_estimates_stft = af_transforms.to_torch_complex(
            self.enc(mixture_from_estimates)
        )  # B x F x N
        estimates_stft = af_transforms.to_torch_complex(
            self.enc(estimates)
        )  # B x 2 x F x N
        bsz, src, freqs, frames = estimates_stft.shape

        estimates_stft = estimates_stft.reshape(bsz * src, freqs, frames)
        mixture_from_estimates_stft = mixture_from_estimates_stft.repeat(
            src, 1, 1
        )  # 2B x F x N
        target_mask = self.get_mask(
            mixture_from_estimates_stft, estimates_stft
        )  # 2B x F x N
        noise_mask = self.get_mask(
            mixture_from_estimates_stft, mixture_from_estimates_stft - estimates_stft
        )  # 2B x F x N
        target_mask = torch.clamp(target_mask, 1e-6, 1e6)
        noise_mask = torch.clamp(noise_mask, 1e-6, 1e6)
        target_mask = target_mask / (2**0.5) + 1j * target_mask / (2**0.5)
        noise_mask = noise_mask / (2**0.5) + 1j * noise_mask / (2**0.5)

        target_scm = beamforming.compute_scm(
            mixture_stft,
            target_mask,
        )  # 2B x C x C x F
        noise_scm = beamforming.compute_scm(mixture_stft, noise_mask)  # 2B x C x C x F
        # clamp not implemented on torch for complex yet
        beamformed = self.beamformer(mixture_stft, target_scm, noise_scm)  # 2B x F x N
        beamformed = af_transforms.from_torch_complex(beamformed)  # 2B x 2F x N
        beamformed = pad_x_to_y(self.dec(beamformed), mixture)
        return beamformed.reshape(bsz, src, samples)
        # bsz, 2, samples


def compute_sa_sdr(mix_features, source_dict, est_sources):
    """
    Return a dict containing "input_sa_sdr" and "sa_sdr", similar to the metrics dict
    returned by get_metrics, for a single session (utterance).
    """
    sources = source_dict["sources"]
    segment_boundaries = source_dict["boundaries"]
    mix = mix_features.unsqueeze(0).repeat((2, 1))
    try:
        metric_dict = {
            "input_sa_sdr": -1
            * optimized_graph_pit_source_aggregated_sdr_loss(
                mix, sources, segment_boundaries
            ).item(),
            "sa_sdr": -1
            * optimized_graph_pit_source_aggregated_sdr_loss(
                est_sources, sources, segment_boundaries
            ).item(),
        }
    except ValueError:
        logging.warning("Error computing metrics for session %s", source_dict["mix_id"])
        metric_dict = {
            "input_sa_sdr": 0,
            "sa_sdr": 0,
        }
    return metric_dict


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = DPRNNTasNet.from_pretrained(model_path)

    # Process long audio file in segments
    continuous_model = LambdaOverlapAdd(
        nnet=model,  # function to apply to each segment.
        n_src=2,  # number of sources in the output of nnet
        window_size=int(
            conf["sample_rate"] * conf["window_size"]
        ),  # Size of segmenting window
        hop_size=int(conf["sample_rate"] * conf["hop_size"]),  # segmentation hop size
        window="hanning",  # Type of the window (see scipy.signal.get_window
        reorder_chunks=True,  # Whether to reorder each consecutive segment.
        enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
    )

    # Handle device placement
    if conf["use_gpu"]:
        continuous_model.cuda()
    model_device = next(continuous_model.parameters()).device

    data_path = (
        os.path.join(conf["test_dir"], "libricss_clean.jsonl")
        if conf["task"] == "sep_clean"
        else os.path.join(conf["test_dir"], "libricss_replayed.jsonl")
    )
    test_set = LibriCSSDataset(
        data_path,
        os.path.join(conf["test_dir"], "libricss_sources.jsonl"),
        sample_rate=conf["sample_rate"],
        multi_channel=True if conf["task"] == "sep_reverb" else False,
    )  # Uses all segment length

    # Sampler and dataloader for the test set
    test_sampler = SimpleCutSampler(test_set.mix, max_cuts=1, shuffle=False)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=None)

    loss_func = GraphPITLossWrapper(assignment_solver="optimal_dynamic_programming")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["decode_dir"], "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(test_set.mix.ids, conf["n_save_ex"])

    if conf["multi_channel"]:
        beam = TimeInvariantBeamWrapper(continuous_model)

    series_list = []
    torch.no_grad().__enter__()
    for batch in tqdm(test_loader):
        inputs, targets = batch
        idx = targets[0]["mix_id"]
        print(f"Processing session: {idx}")

        if inputs.ndim == 2:  # (batch, time)
            inputs = inputs.unsqueeze(1)  # add a fake channel dimension

        # Forward the network on the mixture.
        inputs = tensors_to_device(inputs, device=model_device)

        if conf["multi_channel"]:
            est_sources = beam(inputs)
        else:
            # only use first channel
            est_sources = continuous_model(inputs[:, 0:1, :])

        # print(f"Computing metrics for session: {idx}")
        # utt_metrics = compute_sa_sdr(
        #     inputs[0, 0, :].cpu(),
        #     targets[0],
        #     est_sources[0].cpu(),
        # )

        # series_list.append(pd.Series(utt_metrics))

        # print(f"{idx}: {utt_metrics}")
        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            # Loop over the sources and estimates
            for src_idx, src in enumerate(est_sources[0].cpu().numpy()):
                sf.write(
                    os.path.join(local_save_dir, f"{idx}_{src_idx}.wav"),
                    src,
                    conf["sample_rate"],
                    format="FLAC",
                )
        # Write local metrics to the example folder.
        # with open(local_save_dir + "metrics.json", "w") as f:
        #     json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    # all_metrics_df = pd.DataFrame(series_list)
    # all_metrics_df.to_csv(os.path.join(conf["decode_dir"], "all_metrics.csv"))

    # Print and save summary metrics
    # final_results = {}
    # ldf = all_metrics_df["sa_sdr"] - all_metrics_df["input_sa_sdr"]
    # final_results["sa_sdr"] = all_metrics_df["sa_sdr"].mean()
    # final_results["sa_sdr" + "_imp"] = ldf.mean()
    # print("Overall metrics :")
    # pprint(final_results)
    # with open(os.path.join(conf["decode_dir"], "final_metrics.json"), "w") as f:
    #     json.dump(final_results, f, indent=0)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
