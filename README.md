# Single-Lead ECG-FM Transfer Learning

This repository provides the model construction and training mechanism used in
[Transfer Learning with a Pretrained Foundation Model for Atrial Fibrillation and Flutter Detection on Single-Lead ECG](https://ieeexplore.ieee.org/document/11137575)
([DOI](https://doi.org/10.1109/ITC-CSCC66376.2025.11137575)).

The implementation adapts the pretrained 12-lead ECG-FM encoder to a single
lead, adds a lightweight binary classification head, and exposes the two-stage
fine-tuning procedure described in the paper.

## Scope

Included:

- loading an ECG-FM checkpoint through `fairseq-signals`;
- replacing the first 12-channel convolution with a single-channel convolution;
- initializing that convolution from a selected pretrained lead (Lead II by default);
- temporal mean pooling, dropout, and a binary linear classifier;
- head-only training followed by full-model fine-tuning; and
- class-weighted cross-entropy.

Not included:

- ECG data, labels, or dataset access code;
- preprocessing and dataset-specific split logic;
- pretrained checkpoints;
- evaluation or benchmark scripts; and
- experiment logs, predictions, or performance results.

This is a compact reference implementation of the method, not a packaged
reproduction of the paper's experiments.

## Architecture

The caller supplies a pretrained ECG-FM model. The first convolution is changed
from 12 input channels to one while retaining the selected lead's pretrained
weights. The adapted encoder returns temporal embeddings, which are averaged
over valid time steps and passed through dropout and a two-class linear layer.

By default, `source_lead_index=1` selects Lead II using zero-based indexing.
No fixed input duration is enforced by the model. For the setting described in
the paper, a batch has shape `[batch, 1, 1000]` (10 seconds sampled at 100 Hz).

## Installation

Install PyTorch and the official
[fairseq-signals](https://github.com/Jwoo5/fairseq-signals) implementation first,
then install this package:

```bash
git clone https://github.com/Jwoo5/fairseq-signals.git
pip install -e fairseq-signals
pip install -e .
```

To use the optional checkpoint download helper:

```bash
pip install -e '.[download]'
```

ECG-FM is maintained in the official
[ECG-FM repository](https://github.com/bowang-lab/ecg-fm), and its checkpoint is
available from the
[ECG-FM Hugging Face model card](https://huggingface.co/wanglab/ecg-fm).

## Build the classifier

```python
from single_lead_ecg_fm import build_single_lead_classifier

model = build_single_lead_classifier(
    checkpoint_path="/path/to/mimic_iv_ecg_physionet_pretrained.pt",
    source_lead_index=1,
    num_classes=2,
    dropout=0.3,
)
```

The upstream checkpoint is a PyTorch `.pt` file. Load only checkpoints obtained
from a source you trust.

You can download the official checkpoint without adding it to this repository:

```python
from single_lead_ecg_fm import download_pretrained_checkpoint

checkpoint_path = download_pretrained_checkpoint(cache_dir="./checkpoints")
```

The `checkpoints/` directory and common model-weight extensions are ignored by
Git.

## Two-stage fine-tuning

The training API expects a user-provided iterable whose batches are
`(features, labels)`. Features must be shaped `[batch, 1, time]`; labels are
integer class indices. No dataset implementation is bundled.

```python
from single_lead_ecg_fm import FineTuningConfig, fit_two_stage

config = FineTuningConfig(
    head_epochs=5,
    full_epochs=10,
    head_learning_rate=1e-3,
    full_learning_rate=1e-5,
)

loss_history = fit_two_stage(
    model=model,
    train_loader=train_loader,
    class_counts=[number_of_other_segments, number_of_af_or_afl_segments],
    config=config,
    device="cuda",
)
```

Stage 1 freezes the ECG-FM encoder and optimizes only the classification head.
Stage 2 unfreezes the encoder and fine-tunes the entire model with a lower
learning rate. The returned history contains training loss only; it does not
compute or report performance metrics.

## License and attribution

This repository's code is released under the MIT License. ECG-FM and
fairseq-signals are separate upstream projects; consult their repositories for
their licenses and citation requirements. No upstream model weights are
redistributed here.

## Intended use

This code is provided for research and educational use. It is not a medical
device and is not intended for clinical diagnosis or patient care.
