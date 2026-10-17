# Progress log — COMP3710 Pattern Analysis Project

Project: GFNet vs BaselineCNN on ADNI (Alzheimer's vs Cognitively Normal)
Folder: `recognition/GFNet_s4967283/`
Branch: `topic-recognition`

---

## Day 1 — Tuesday 13 October

### What I decided

Picked **GFNet on the ADNI dataset**, classifying brain MRI slices as AD
(Alzheimer's Disease) or NC (Cognitively Normal). This is a Hard difficulty
option, which matters because Easy caps the implementation marks at 10/20 and
Normal at 15/20. Hard is the only tier that leaves enough headroom for a 7.

Ruled out anything on the OASIS dataset, because I already did a UNet and a VAE
on OASIS for Demo 2 and the spec says the final model has to be substantially
different from previous work.

### What I built

**`audit_adni.py`** — a script to inspect the dataset before writing any loader.
I wrote this first because the data loader has to split by patient, and I could
not do that until I knew whether patient IDs were even recoverable from the file
names.

**`modules.py`** — the baseline CNN. Four conv blocks (conv, batch norm, ReLU,
max pool) growing 1 → 32 → 64 → 128 → 256 channels, then global average pooling
and a small fully connected head. 421,474 trainable parameters.

Two choices worth remembering:
- Global average pooling instead of flattening a fixed grid, so the model works
  at any input resolution. Verified by running it at both 224 and 256.
- Two output logits rather than one sigmoid, so I get a proper probability per
  class after softmax. I need calibrated confidence later for the uncertainty
  investigation, and this keeps the baseline and GFNet handling identical.

**`dataset.py`** — the data loader, with patient-level splitting.

**`.gitignore`** — blocks `__pycache__`, vim swap files, model weights and
dataset files. The spec says committing large files gets the pull request
rejected, so `*.pt` and the dataset patterns matter more than the editor junk.

### What the audit found

```
train/AD   10400 images   520 patients
train/NC   11120 images   556 patients
test/AD     4460 images   223 patients
test/NC     4540 images   227 patients
```

- File names are `<patientID>_<slice>.jpeg`, e.g. `1003730_100.jpeg`
- Every patient contributes exactly 20 slices
- Images are 256 wide × 240 high, greyscale, pixel values 0–255
- No patient appears in both train and test, so the shipped split is clean

### The leakage problem, and what I did about it

Each patient has 20 slices cut from the same 3D scan, so those images look very
similar to one another. A random image-level split would scatter one person's
slices across training and validation. The model could then recognise the
patient instead of the disease, validation accuracy would look great for the
wrong reason, and the test score would collapse with no obvious explanation.

So every split happens at the patient level: I choose which patients go where
first, then all 20 of their slices follow. `check_no_patient_overlap` raises an
`AssertionError` if a patient ever lands in two splits. It crashes rather than
warns, because if that fires every number downstream is worthless.

I also split the two classes separately, so validation gets a fair share of each
rather than whatever the shuffle happened to produce.

### Splits I ended up with

| Split      | Images | Patients | NC    | AD    |
|------------|--------|----------|-------|-------|
| train      | 18300  | 915      | 9460  | 8840  |
| validation | 3220   | 161      | 1660  | 1560  |
| test       | 9000   | 450      | 4540  | 4460  |

Validation is 15.0% of the training patients. Stratification held up: 78 of 520
AD patients and 83 of 556 NC patients, both within a tenth of a percent of 15%.

Class balance overall is NC 51.7% / AD 48.3%. Mild enough that I do not need
weighted loss or resampling — but I checked rather than assumed.

### Normalisation

Measured over 2000 training images (training split only, so nothing leaks in
from validation or test):

```
PIXEL_MEAN = 0.1159
PIXEL_STD  = 0.2236
```

The mean is low because these slices are mostly black background. With the
placeholder 0.5/0.5 the normalised range was −1.000 to 0.953, which squashed the
actual brain tissue into a narrow band. With the measured values the range is
−0.518 to 3.849: background sits near −0.52 and tissue spreads across the
positive range. The asymmetry reflects the data honestly.

### Other design decisions I need to be able to defend

**Resize 256×240 → 224×224.** GFNet needs a fixed square input. I resize rather
than crop, because cropping could cut off the outer cortex, which is exactly the
tissue that atrophies in Alzheimer's. The stretch is identical for every image so
the model sees a consistent world.

**Horizontal flip augmentation.** The brain is roughly symmetric and these slices
are not labelled by side, so a mirrored brain is still plausible. Honest caveat:
Alzheimer's atrophy is not always symmetric, so this does throw away a little
real signal. I kept it for the extra variety, but it is a decision to test rather
than assume.

**Small rotation and shift (5°, 5%).** Patients are not positioned identically in
the scanner, so this is variation the model will genuinely meet. Kept small
because a heavily rotated slice is not realistic.

### Mistake I made

Committed my first two commits to `main` instead of `topic-recognition`. Fixed by
cherry-picking them onto the right branch rather than resetting history — the
commit log is a pass hurdle and deleting commits would have been worse than the
original mistake. Left the duplicates on `main`, since nobody marks that branch.

Added a habit: run `git branch --show-current` at the start of every session.

---

## Day 2 — Wednesday 14 October

### The main job: the feasibility review

The midpoint check-off is an ungraded hurdle. If I do not complete it I cannot
pass the assessment at all, whatever the code looks like. So the whole evening
went on the one-page written review for Thursday morning.

It covers the four things the spec asks for:

1. **User need and acceptance criteria** — a clinician doing early screening,
   five testable criteria including a resource budget of 16 GB and 2 hours.
2. **Model choice** — GFNet, Hard tier, justified against the Fourier material
   and my own Demo 2 Part 1 work where I measured naive DFT at 34.2 s against
   FFT at 0.13 ms for N=4096.
3. **Preliminary evidence** — the data audit from Day 1, the splits, and the
   measured normalisation constants.
4. **Risks and fallback** — GFNet's FFT block is the part I am least sure of;
   fallback is ConvNeXt on the same dataset, same dilemma, same evaluation code.

I framed the whole thing around the confidence dilemma rather than around
accuracy, because that is what the spec actually asks me to investigate.

### What I built

**`train.py`** — the training loop. Validates every epoch, saves the best
checkpoint on validation accuracy rather than the last epoch, plots loss and
metric curves, and records peak GPU memory, timing and parameter count.

The important design choice: it saves `test_predictions_<model>.npz` with the
per-image probabilities, true labels and patient IDs. That means the whole
calibration investigation can run later without retraining anything, and the
patient IDs let me tell apart "five mistakes from one patient" and "five
mistakes from five patients" — completely different failure modes.

**`run_smoke.sh`** — SLURM script for a short 3-epoch test.

### Mistakes I made

**Wrong partition.** I first wrote `--partition=a100`. My Demo 2 notes have two
findings about this and I used the stale one. The early note said the comp3710
partition rejects jobs; the later note said the rejection was caused by a
missing `--account=comp3710`. The right answer is `--partition=comp3710`, which
is the partition allocated to this course. The a100 partition shares nodes with
cosc3500 and a100-grind, so jobs sit pending behind other courses.

I now have the reasoning for every SLURM flag written into `run_smoke.sh` as
comments, so neither I nor anyone reading the repo has to rediscover it.

**Commits that did not land.** I pasted several `git add` and `git commit` pairs
for the same file. Only the last one took effect. Staging the same file twice
leaves nothing for the second commit. One commit per logical change, one file at
a time.

**Local environment.** My Windows machine had CPU-only `torch` with a CUDA
`torchvision`, which made `import torchvision` fail outright. Fixed with a
force-reinstall of both from the cu128 index in one command. Rule: install torch
and torchvision together, never separately.

---

## Day 3 — Thursday 15 October

### The check-off is done

Hurdle cleared. The tutor read the review, gave brief feedback and recorded
completion. Their one warning was about data leakage.

I had already handled the main case with patient-level splitting. But there is
one thing I cannot verify from the preprocessed data: I assume the number before
the underscore is a patient ID rather than a scan or session ID. ADNI is a
longitudinal study, so the same person can be scanned more than once. I asked
and the tutor confirmed they are patient IDs.

### The smoke test worked

```
epoch 1: 75.4s    epoch 2: 16.1s    epoch 3: 15.4s
test accuracy 0.7081, peak VRAM 2.32 GB
```

Epoch 1 is slow because of cuDNN algorithm selection and file caching. Steady
state is about 16 s per epoch, so a 30-epoch run takes under 10 minutes. My
feasibility review estimated 6 to 8 A100-hours. The real figure is a small
fraction of that, which means I can afford multiple seeds and ablations.

This also closed risk 2 from the feasibility review: `torch 2.14.0+cu130`,
`cuda available: True`, `NVIDIA A100-PCIE-40GB`.

### What I built

**`visualise_data.py`** — four figures of the actual data, plus a measured
within-patient against between-patient correlation. Figure 2 shows all 20 slices
of one patient side by side, which is the visual proof of why the split must be
by patient.

**GFNet in `modules.py`** — patch embedding, the global filter block, the
feed-forward block, position embeddings and a mean-pooled classifier head.
4,788,482 parameters, which is 11.4 times the baseline.

### The test that mattered

Six self-tests, all passing. The important one sets every filter weight to
exactly 1 + 0i. FFT, multiply by one, inverse FFT should return the input
unchanged. Largest difference: **4.768e-07**, which is float32 rounding error.

That proves the transform plumbing is right — the dimensions, `norm="ortho"`,
the `s=` argument on the inverse, and the half-spectrum width. The rfft output
is 14 × 8, not 14 × 14, because a real input has a symmetric spectrum. Getting
that wrong gives either a shape error or a silent broadcast that trains badly
for no visible reason.

Risk 1 from the feasibility review is now closed, and it cost no GPU time.

---

## Day 4 — Friday 16 October

I lost Thursday evening and most of Friday. Caught up in one long session.

### What I built

- Wired gfnet into `train.py` through a shared `build_model` in `modules.py`
- `run_train.sh` — one script that takes model, seed and epochs as arguments
- `evaluate_calibration.py` — ECE, reliability diagrams, the reject-option
  threshold sweep, and patient-level aggregation
- `brightness_baseline.py` — a deliberately stupid classifier, see below

### First real results (30 epochs, learning rate 3e-4)

| Model | Test accuracy | Macro F1 | AD recall |
|---|---|---|---|
| Baseline | 0.7241 | 0.7138 | 0.5390 |
| GFNet | 0.6890 | 0.6749 | 0.4848 |

GFNet lost. That is a valid answer to the spec's question about whether the
extra complexity is justified, not a failure.

### The finding that matters most

Both models prefer to say "normal". The baseline misses 46% of AD patients. The
GFNet misses 52%. For a screening tool that is the worst possible failure mode,
because a missed case gets no follow-up and nobody ever checks.

AD precision is high (0.85, 0.81). So when a model says AD it is usually right.
The problem is that it rarely says AD.

### Patient aggregation does not rescue it

I expected a large gain from averaging each patient's 20 slices. I got +1.1 and
+2.0 points. The reason: 51 patients have every single one of their 20 slices
wrong. Averaging cannot fix a patient the model misreads 20 times out of 20.
The errors are patient-systematic, not slice-random.

### Both models are confidently wrong

Mean confidence 0.929 when right, 0.878 when wrong. Separation 0.051. ECE 0.191.

The spec asks whether the network "remains stubbornly overconfident even when
making incorrect predictions". Mine does, and now I have the number.

### A lower threshold is nearly free

At an AD threshold of 0.10 instead of 0.50, the baseline misses 1,383 AD slices
instead of 2,056. That is 673 fewer missed cases, and accuracy goes slightly
**up**, from 0.7241 to 0.7382.

So the default 0.5 is the wrong operating point for a screening tool. But the
threshold only moves errors between classes. It does not add discriminative
power, and accuracy stays near 0.73 at every threshold.

### The brightness experiment

The intensity histograms showed AD tissue is darker than NC tissue. That raised
a question I had to answer: does my model read anatomy, or does it read
brightness?

So I built a classifier that uses one number per slice — the mean intensity of
its tissue pixels — and fitted the threshold on the training split only.

```
Brightness rule:  0.585
GFNet:            0.689
BaselineCNN:      0.724
```

The CNN adds 13.9 points over the brightness rule, so the networks learn real
structure. But 0.585 is above chance, so about 8.5 points of the signal is
brightness. Both numbers go in the report.

I also checked the train and test intensity histograms against each other. They
look the same, so a distribution shift between the shipped folders is not the
explanation for the validation-to-test gap.

---

## Day 5 — Saturday 17 October

### Results after 80 epochs

| Model | Val | Test slice | Test patient | Best epoch |
|---|---|---|---|---|
| Baseline | 0.9326 | 0.7360 | 0.7422 | 75 of 80 |
| GFNet | 0.8134 | 0.7106 | 0.7422 | 54 of 80 |

Doubling the training bought 1.2 points for the baseline and 2.2 for GFNet.
GFNet reached its best at epoch 54 and did not improve after. It has converged.

### The gap is overfitting, not a data problem

| Baseline | Validation | Test | Gap |
|---|---|---|---|
| 30 epochs | 0.8894 | 0.7241 | 0.165 |
| 80 epochs | 0.9326 | 0.7360 | 0.197 |

Validation rose 4.3 points. Test rose 1.2. The gap got wider. The model fits the
training cohort better and better, and the test cohort does not follow.

### Longer training made the confidence worse

| Baseline | ECE | Confidence when wrong | Separation |
|---|---|---|---|
| 30 epochs | 0.191 | 0.878 | 0.051 |
| 80 epochs | 0.216 | 0.922 | 0.041 |

More certain, not more correct. A wrong prediction now claims 92% certainty.

### The two models tie at patient level

Both get exactly 334 of 450 patients right. But the failure shapes differ:

| Model | All 20 slices right | All 20 slices wrong |
|---|---|---|
| Baseline | 246 | 60 |
| GFNet | 159 | 35 |

GFNet has 42% fewer catastrophic patients. A patient wrong on all 20 slices is
one the system misclassifies with no internal disagreement to flag it.

### Temperature scaling fixes GFNet but not the baseline

| Model | Temperature | ECE before | ECE after | Change |
|---|---|---|---|---|
| Baseline | 1.36 | 0.2164 | 0.1963 | −9% |
| GFNet | 1.90 | 0.1819 | 0.1102 | −39% |

Accuracy did not move for either model, which is the expected behaviour and
confirms the implementation is right.

GFNet's overconfidence is a single global scaling problem, and one parameter
fixes most of it. The baseline's is structural, so the same fix barely helps.

I fitted the temperature on the validation split only. Fitting it on test and
then reporting test calibration would be tuning on the test set, which is the
same sin the patient-level splitting exists to avoid.

### What I built

**`predict.py`** — the fifth required file. Loads a checkpoint, runs inference,
prints a confusion matrix and per-class metrics, saves the probabilities, and
draws three figures: example predictions, the most confident mistakes for the
error autopsy, and the learned GFNet frequency filters.

### Where the recommendation stands

| Measure | Baseline | GFNet |
|---|---|---|
| Slice accuracy | **0.7360** | 0.7106 |
| Patient accuracy | 0.7422 | 0.7422 |
| Catastrophic patients | 60 | **35** |
| ECE raw | 0.2164 | **0.1819** |
| ECE after scaling | 0.1963 | **0.1102** |
| Parameters | **421,474** | 4,788,482 |
| Inference | **12,263 img/s** | 4,640 img/s |
| Peak memory | 2.32 GB | **1.58 GB** |

The baseline is the better classifier. GFNet is the more trustworthy one. For a
triage system whose job is to route uncertain cases to a specialist, trustworthy
confidence is the property that matters.

### Mistakes today

1. Ran `predict.py` on the login node. It used CPU and four worker processes on
   a shared machine. Rule I am keeping: a GPU, any `num_workers` above zero, or
   anything over a few minutes goes to SLURM.
2. A stray `)` in `train.py` killed a job after 7 seconds. I now run
   `python -c "import ast; ast.parse(open('train.py').read())"` after every edit,
   before submitting anything.
3. An `srun` session timed out unused and wasted a 20-minute GPU allocation. Use
   a realistic `--time`, and `exit` when finished.

---

## Still open

- **The 0.8 target.** I am at 0.736 slice and 0.742 patient. More epochs will
  not close it. Asking the tutor on Monday whether the figure is firm, and
  showing the investigation.
- **Strong-augmentation run** is queued. It tests the overfitting theory with
  wider affine transforms, brightness jitter and weight decay at 1e-3. The
  brightness jitter is the key part, because the brightness experiment showed a
  mean-intensity rule alone reaches 0.585.
- **Error autopsy write-up** from the failure figures. I have the patient IDs
  and confidences from `predict.py`.
- **`README.md`** is the only required file still missing. It carries 10 marks.

## Things I should be able to explain out loud

- Why patient-level splitting matters, and what would go wrong without it
- Why the measured normalisation constants changed the pixel range the way they did
- What GFNet does instead of self-attention, and why the convolution theorem
  makes that equivalent to global mixing
- Why the rfft filter is 14 × 8 and not 14 × 14
- Why temperature scaling must be fitted on validation and not on test
- Why accuracy does not change when the temperature changes
- Why the widening validation-to-test gap means overfitting rather than a data shift
- Why patient-level accuracy is the clinically honest number
- Why 60 patients wrong on all 20 slices is worse than 60 scattered wrong slices
- Why I ran a brightness-only classifier, and what its 0.585 tells me
