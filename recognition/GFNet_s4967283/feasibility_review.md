# Feasibility Review — GFNet for Alzheimer's Classification on ADNI

**Manav Sachdeva (s4967283)** · COMP3710 Pattern Analysis Project · 14 October 2026
Project 2.1 Classification · Model: GFNet · Difficulty: **Hard**

---

## 1. User need, scope and acceptance criteria

**Intended user.** A clinician doing early Alzheimer's screening. The prototype is not
meant to replace their judgement. It sorts incoming structural brain MRI scans into
cases it is confident about and cases it is not, so the uncertain ones get a
specialist's attention first.

**Purpose.** Classify a 2D T1-weighted brain MRI slice as AD (Alzheimer's Disease) or
NC (Cognitively Normal), and attach a confidence score to each prediction that can
actually be trusted.

**The dilemma I am investigating.** A model can reach high overall accuracy while being
confidently wrong on the borderline cases — which are exactly the ones a clinician needs
flagged. A false negative on early Alzheimer's is far more costly than a false positive.
So the real question is not "how accurate is it" but "does its confidence mean anything".

**Acceptance criteria** (all measured on the held-out test split):

| # | Criterion | Target |
|---|-----------|--------|
| 1 | GFNet test accuracy, AD vs NC | ≥ 0.80 |
| 2 | Baseline CNN trained and evaluated on identical splits | reported alongside |
| 3 | Expected Calibration Error (ECE) computed for both models | GFNet ECE compared against baseline |
| 4 | Reject-option rule: accuracy on cases the model keeps, after routing low-confidence cases to a human | ≥ 0.95 retained accuracy, with the referral rate reported |
| 5 | Resource budget: peak GPU memory and training wall-clock on one A100 | ≤ 16 GB, ≤ 2 hours |

Criterion 4 is the one that answers the dilemma. Criteria 1 and 2 only establish that
there is something worth analysing.

## 2. Model choice and course concepts

**Dataset.** ADNI from `/home/groups/comp3710/ADNI/AD_NC`. Two classes, 2D greyscale
slices, 30,520 images total.

**Model.** GFNet (Global Filter Network), Rao et al., 2023 — a Hard-difficulty option.

A Vision Transformer mixes information between image patches using self-attention,
which costs roughly n² operations for n patches. GFNet keeps the global mixing but
replaces attention with three steps: take the 2D FFT of the feature map, multiply by a
filter whose weights are learned, then inverse FFT back. By the convolution theorem,
multiplying in the frequency domain is the same as convolving across the whole image in
normal space, so one multiplication still lets every patch see every other patch — at
n log n instead of n².

**Why this fits the problem.** Alzheimer's shows up as large-scale structural change:
the hippocampus shrinks, the ventricles enlarge, the cortex thins. Those are
low-frequency features. A model whose learned parameters are literally frequency weights
is a natural match, and I can plot which frequencies it learned to weight heavily, which
gives me a figure that explains the model rather than just reporting its score.

**Course connection.** This builds directly on the Fourier and convolution material and
on my own Demo 2 Part 1 work, where I implemented the DFT naively and compared it to the
FFT (34.2 s versus 0.13 ms at N=4096). The speed argument for GFNet over attention is the
same argument, one level up.

**Baseline.** A plain four-block CNN I wrote myself (421,474 parameters). Deliberately
not tuned hard, so it stays an honest reference point rather than a strawman.

## 3. Preliminary feasibility evidence

I audited the dataset before writing any model code, because the splitting strategy
depended on what I found.

```
train/AD  10400 images, 520 patients      test/AD  4460 images, 223 patients
train/NC  11120 images, 556 patients      test/NC  4540 images, 227 patients
```

- File names are `<patientID>_<slice>.jpeg`. Every patient contributes exactly 20 slices.
- Images are 256 × 240, greyscale, 0–255.
- No patient appears in both the shipped train and test folders.

**The leakage risk I found and handled.** Because each patient has 20 near-identical
slices from one scan, a random image-level split would scatter one person's slices across
training and validation. The model could then recognise the patient rather than the
disease, and validation accuracy would look excellent for entirely the wrong reason. So
every split happens at the patient level, and an assertion raises an error if a patient
ID ever appears in two splits.

Splits I built (15% of training patients held back for validation, stratified by class):

| Split | Images | Patients | NC | AD |
|-------|--------|----------|-----|-----|
| train | 18,300 | 915 | 9,460 | 8,840 |
| validation | 3,220 | 161 | 1,660 | 1,560 |
| test | 9,000 | 450 | 4,540 | 4,460 |

Class balance is NC 51.7% / AD 48.3% — mild enough that weighted loss is not needed, but
I checked rather than assumed.

**Normalisation.** Measured over 2,000 training images only: mean 0.1159, std 0.2236. The
mean is low because these slices are mostly black background. Using placeholder values of
0.5 compressed the actual brain tissue into a narrow band; the measured values spread it
properly.

**Verified so far.** Data loader runs end to end and produces correct `(batch, 1, 224, 224)`
tensors. Baseline model forward pass verified at two input resolutions.

## 4. Risks, compute budget, next experiment, fallback

**Risks.**

1. *GFNet implementation complexity.* The FFT filter block is the part I am least certain
   about — getting the complex tensor shapes and the learnable filter dimensions right.
   I plan to unit-test the block in isolation before putting it in a training loop.
2. *Environment.* Training must run on the a100 partition with `--account=comp3710` and
   `gpu:a100:1`, and no `--mem` request. I know these constraints from Demo 2. I still
   need to confirm my torch build initialises CUDA on an a100 node.
3. *Overfitting.* 915 training patients is not many for a model of this class. Mitigations
   in place: augmentation, dropout, and checkpointing on validation rather than training loss.

**Compute budget.** Roughly 6–8 A100-hours total: about 1 hour for the baseline, 2 hours
for GFNet, and the rest for reruns and the calibration experiments. Short, honest
`--time` values, since the 3-hour partition default blocks backfill and lands me further
back in the queue.

**Next experiment.** A 3-epoch baseline smoke test on one A100 to confirm the loop trains
and to get a real measurement of time per epoch and peak VRAM. That number sets whether
the 2-hour budget in criterion 5 is realistic.

**Fallback.** If the GFNet filter block cannot be made to train reliably, I drop to
ConvNeXt on the same dataset — also Hard difficulty, same dilemma, same evaluation code,
more conventional architecture. Everything in sections 1 and 3 stays valid, so the
switch costs me the model file and nothing else.