# Progress log — COMP3710 Pattern Analysis Project

Project: GFNet on ADNI (Alzheimer's vs Cognitively Normal classification)
Folder: `recognition/GFNet_s4967283/`
Branch: `topic-recognition`

---

## Day 1 — Tuesday 14 October

### What I decided

Picked **GFNet on the ADNI dataset**, classifying brain MRI slices as AD
(Alzheimer's Disease) or NC (Cognitively Normal). This is a Hard difficulty
option, which matters because Easy caps the implementation marks at 10/20 and
Normal at 15/20. Hard is the only tier that leaves enough headroom.

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
than assume. Worth an ablation if I have spare GPU time.

**Small rotation and shift (5°, 5%).** Patients are not positioned identically in
the scanner, so this is variation the model will genuinely meet. Kept small
because a heavily rotated slice is not realistic.

### Mistake I made

Committed my first two commits to `main` instead of `topic-recognition`. Fixed by
cherry-picking them onto the right branch rather than resetting history — the
commit log is a pass hurdle and deleting commits would have been worse than the
original mistake. Left the duplicates on `main`, since nobody marks that branch.

Added a habit: run `git branch --show-current` at the start of every session.

### Open / next

- **Check the GPU environment works** before writing more training code. The
  login node reports the NVIDIA driver as too old, which is expected there, but I
  have not confirmed the `comp3710` venv's torch build works on the a100 nodes.
- Write `train.py` and run a baseline smoke test on Rangpur.
- Write the one-page feasibility review for Thursday's check-off. That is an
  ungraded hurdle but I cannot pass the assessment without it.

### Things I should be able to explain out loud

- Why patient-level splitting matters, and what would go wrong without it
- Why the measured normalisation constants changed the pixel range the way they did
- What GFNet does instead of self-attention, and why the convolution theorem
  makes that equivalent to global mixing
- Why I chose global average pooling over a flattened head
