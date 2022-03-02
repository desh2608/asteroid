#!/bin/bash
. ./path.sh

# Exit on error
set -e
set -o pipefail

# Corpus dir
librispeech_dir=/export/corpora5/LibriSpeech
librispeech_ctm=data/librispeech_ctm
libricss_dir=/export/c01/corpora6/LibriCSS

# RIRs
rir_dir=data/rirs

# Pretrained model to initialize the DPRNN
init_model=../../whamr/DPRNN/exp/train_dprnn_v4/best_model.pth

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
decode_tag=""  # Controls the directory name associated to the decoding
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
ngpu=1

# Data
sample_rate=16000
data_dir=data  # Local data directory (No disk space needed)
task=sep_clean  # Task to train (sep_clean, sep_reverb)

# Training
batch_size=50  # Batch size (in seconds)
num_workers=8
optimizer=adam
lr=0.0001
epochs=50
weight_decay=0.00001

# Architecture config
# kernel_size=2
# stride=8
# chunk_size=100
# hop_size=50

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
  echo "Stage 0: Create Lhotse manifests for train and dev mixtures (creates both clean and reverberant versions)"
  python local/create_meetings.py --corpus-dir $librispeech_dir \
    --ctm-dir $librispeech_ctm --rir-dir $rir_dir \
    --output-dir data/librispeech
fi

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Prepare Lhotse manifests for evaluation (clean and replayed LibriCSS)"
  python local/prepare_libricss_data.py --corpus-dir $libricss_dir --output-dir data/libricss
fi

expdir=exp/train_dprnn_${tag}
echo "Results from the following experiment will be stored in $expdir"

if [[ $task == "sep_clean" ]]; then
  task_affix=_clean
else
  task_affix=_reverb
fi

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  queue-freegpu.pl --mem 8G --gpu 4 --config conf/gpu.conf $expdir/train.log \
    python train.py \
    --train_mix data/librispeech/train-clean-100_mixed${task_affix}.jsonl \
    --train_sources data/librispeech/train-clean-100_sources.jsonl \
    --valid_mix data/librispeech/dev-clean_mixed${task_affix}.jsonl \
    --valid_sources data/librispeech/dev-clean_sources.jsonl \
    --nondefault_nsrc 3 \
    --noise_src true \
    --mixture_consistency true \
    --sample_rate $sample_rate \
    --lr $lr \
    --epochs $epochs \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --weight_decay $weight_decay \
    --exp_dir ${expdir}/ \
    --init-model $init_model
fi

if [[ $stage -le 4 ]]; then
  echo "Stage 4 : Evaluation"
  queue-freegpu.pl --mem 8G --gpu 1 --config conf/gpu.conf $expdir/decode_${decode_tag}/decode.log \
    python eval.py \
    --task $task \
    --test_dir data/libricss \
    --use_gpu $eval_use_gpu \
    --exp_dir ${expdir} \
    --window-size 8 \
    --hop-size 2 \
    --decode_dir ${expdir}/decode_${decode_tag} \
    --multi-channel
fi
