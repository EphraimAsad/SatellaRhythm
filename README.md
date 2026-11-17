# SatellaRhythm-Edge v0.3

**Lightweight single-lead ECG neural-network research project — not a clinical product.**

> **Warning:** The model is experimental and **failed its predefined research gate**.
> Do not use it to diagnose, screen, triage, monitor or reassure patients. No clinical safety
> or cross-device generalisation has been demonstrated.

**Creator:** Zain Asad (`EphraimAsad`)  
**Model and detailed model card:** https://huggingface.co/EphAsad/SatellaRhythm-Edge-v0.3

## What is here?

A reproducible 43-cell, output-stripped Google Colab training notebook; a compact
2-head 1D ResNet for rhythm and ECG signal quality; an inference wrapper using CPU ONNX;
and the public, quantified limitations. **Weights are hosted on Hugging Face, not GitHub**.
No patient recordings, original dataset copies, executable outputs, secrets or logs are committed.

## Get started with an ECG waveform (research only)

```bash
python -m pip install -r requirements-inference.txt
```

```python
import numpy as np
from huggingface_hub import snapshot_download
from infer_onnx import RhythmEdge

folder = snapshot_download(
    repo_id="EphAsad/SatellaRhythm-Edge-v0.3",
    allow_patterns=["satellarhythm_edge_v03.onnx", "config_research.json"],
)
model = RhythmEdge(folder)
# Use only lawfully obtained, appropriately protected, single-lead ECG waveform data.
# This `.npy` path is a user-provided example, not a bundled recording.
waveform = np.load("path/to/your_ecg_waveform.npy", allow_pickle=False)
print(model.predict(waveform, sample_rate_hz=300))
```

The `RhythmEdge` wrapper converts sampling rates, makes 10-second windows, applies the
training-time robust scaling, runs the 2-output ONNX model and performs a research-only
rejection policy. It does not accept PPG, raw heart-rate values, ECG screenshots or multi-lead
images. Outputs are **not** medically calibrated and a `NORMAL` result is not reassuring.

## Reproduce training

Open `train/SatellaRhythm_Edge_v03_Training.ipynb` in Google Colab and run on a GPU.
The notebook is **unexecuted** in this repository to avoid publishing notebook outputs,
Colab account metadata and individual record examples. The training notebook downloads
PhysioNet Challenge 2017 and 2011 data directly from their source pages;
review their ODC-By 1.0 terms and citations before downloading or using the datasets.

The training notebook uses two-stage weighted training and validation-selected rejection.
Its final evaluation is record-disjoint only, **not confirmed patient-disjoint**. It includes
an optional AFDB training extension, which was **disabled** for this checkpoint.

## Recorded result (v0.3)

| Research-only metric | Internal held-out result |
|---|---:|
| Four-class macro F1 | 0.7027 |
| AF sensitivity | 0.9115 |
| AF precision | 0.7518 |
| Other rhythm sensitivity | 0.6529 |
| Real noisy ECG recall | 0.8333 |
| Abnormal rhythms incorrectly reported NORMAL | **0.1597** |
| Four-class 2017 recording count | 1,279 |
| Research gate | **NOT MET** |

Two failed targets: macro F1 should have been at least 0.72, and abnormal-to-normal
rate no more than 0.15. Dataset limitations, confidence details, confusion matrix,
non-independence and hardware timing are documented in the [model card](https://huggingface.co/EphAsad/SatellaRhythm-Edge-v0.3).

## Repository contents

| File | Description |
|---|---|
| `train/SatellaRhythm_Edge_v03_Training.ipynb` | Training and internal evaluation; stripped of outputs |
| `infer_onnx.py` | CPU inference wrapper; raw ECG preprocessing |
| `requirements-inference.txt` | Inference dependencies |
| `requirements-training.txt` | Training dependencies |
| `DISCLAIMER.md` | Safety and intended-use restrictions |

## Rights and provenance

Source datasets: [PhysioNet Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/)
and [Challenge 2011](https://physionet.org/content/challenge-2011/1.0.0/).
Both identify ODC-By 1.0 dataset terms; cite original contributors and PhysioNet.
No licence is asserted for model weights or other third-party materials by this repository.
The rights to reuse the code should be reviewed separately from the datasets and weights.

**No medical or diagnostic use.**
