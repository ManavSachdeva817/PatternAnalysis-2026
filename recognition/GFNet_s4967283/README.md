# GFNet for Alzheimer's Classification on ADNI Brain MRI

**Manav Sachdeva (s4967283).** COMP3710 Pattern Analysis Project.
Project 2.1 Classification. Model: GFNet. Difficulty: Hard.

---

## The problem

Alzheimer's disease destroys brain tissue. The damage shows up on a structural
MRI scan. The hippocampus gets smaller. The ventricles get larger and fill the
space. The cortex gets thinner. If a computer can find those changes early, a
clinician can start treatment sooner.

This project reads one 2D slice of a T1-weighted brain MRI. It puts the slice
in one of two classes: AD (Alzheimer's Disease) or NC (Cognitively Normal). The
data comes from the ADNI study. ADNI is a North American research programme
that has collected brain scans since 2004.

The hard part is not the difference between a sick brain and a healthy brain.
The hard part is that healthy brains differ a lot from person to person.
Healthy brains also get smaller with normal age. The model must separate "this
brain is sick" from "this brain belongs to a different person".

### The question this report answers

The assignment asks for more than accuracy. The open research question for this
field is about confidence.

> Does the confidence of the model match whether the model is correct? Or does
> the model stay confident when it is wrong? Can a confidence threshold
> automate the easy cases and send the unsure cases to a specialist?

A screening tool with 74% accuracy that knows when it is unsure is more useful
than one with 74% accuracy that always sounds certain. This report measures the
confidence of both models. It then says plainly whether you can trust that
confidence.

---

## How GFNet works

A Vision Transformer cuts an image into patches. It then uses self-attention,
so every patch can exchange information with every other patch. That global mix
is the strength of a transformer. But attention costs about n² operations for n
patches.

GFNet keeps the global mix and removes the attention. Each block does three
steps:

1. Take the 2D real FFT of the token grid. The result is complex.
2. Multiply the result by a learnable complex filter of the same shape. These
   weights are the parameters that the model learns. They are frequency gains.
3. Take the inverse FFT back to normal space.

The convolution theorem says that a multiply in the frequency domain equals a
convolution across the whole grid in normal space. So one multiply lets every
patch see every other patch. The cost is n log n instead of n².

### Why this architecture suits this disease

Alzheimer's atrophy is large-scale change in structure. The ventricles get
larger, the cortex gets thinner, the hippocampus gets smaller. These are
low-frequency features.

A model whose parameters are frequency weights is a good match. You can also
look inside it. `predict.py` plots the magnitude of each learned filter, so
this report can show which frequencies the model chose. A convolutional network
gives no equivalent picture.

### One detail that is easy to get wrong

A real-valued input has a symmetric spectrum. `torch.fft.rfft2` uses that fact
and returns only half of the frequencies. For a 14 × 14 token grid the output is
14 × **8**, not 14 × 14.

The learnable filter must match that shape. If the shape is wrong you get a
shape error, or worse, a silent broadcast that trains badly for no visible
reason. `modules.py` holds a self-test for this. The test sets every filter
weight to 1 + 0i and checks that the block returns its input without change. It
passes to within 4.8e-07, which is float32 rounding error.

![GFNet learned frequency filters](figures/filters_gfnet.png)

*Each panel is the learned filter magnitude of one block. Top-left is low
frequency, which means large structures. Bottom-right is high frequency, which
means fine texture.*

---

## Feasibility review

*This is the one-page review I presented at the midpoint check-off, as Section 3
of the assignment specification requires.*

**Intended user.** A clinician who does early Alzheimer's screening. The
prototype does not replace the judgement of that clinician. It sorts scans into
two groups: cases the model is sure about, and cases it is not. The unsure cases
reach a specialist first.

**Purpose.** Put a 2D T1-weighted brain MRI slice in the AD class or the NC
class, and attach a confidence score that you can trust.

**Acceptance criteria.** I measured all of them on the held-out test split.

| # | Criterion | Target | Outcome |
|---|-----------|--------|---------|
| 1 | GFNet test accuracy, AD against NC | ≥ 0.80 | **Not met.** See Limitations |
| 2 | Baseline CNN trained on identical splits | reported | Met |
| 3 | Expected Calibration Error for both models | reported | Met |
| 4 | Reject-option rule that reaches ≥ 0.95 retained accuracy | ≤ 30% referral | **Not achievable.** See Calibration |
| 5 | Peak GPU memory and training time on one A100 | ≤ 16 GB, ≤ 2 h | Met. 2.32 GB, 18 min |

**Model choice.** GFNet (Rao et al., 2023) is a Hard-difficulty option. The
choice builds on the Fourier material in this course and on my own Demo 2 Part 1
work. There I measured a naive DFT at 34.2 s against an FFT at 0.13 ms for
N = 4096. The argument for GFNet against attention is the same argument one
level up.

**Baseline.** A plain four-block CNN written for this project, with 421,474
parameters. I did not tune it hard, so it stays an honest reference point and
not a weak target.

**Risks I found, and what happened.**

1. *Complexity of the GFNet filter block.* Closed by the FFT identity
   self-test, before I spent any GPU time.
2. *Cluster environment.* Confirmed to work. torch 2.14.0+cu130 on an
   A100-PCIE-40GB.
3. *Overfit on 915 training patients.* This risk happened. See the overfit
   section below for the diagnosis and the fix.

---

## Repository contents

| File | Purpose |
|------|---------|
| `modules.py` | `BaselineCNN` and `GFNet` definitions, plus six self-tests |
| `dataset.py` | Data loader with patient-level splits and augmentation |
| `train.py` | Training loop, validation, checkpoints, metric plots |
| `predict.py` | Loads a checkpoint, runs inference, draws the report figures |
| `audit_adni.py` | Dataset inspection, written before the loader |
| `visualise_data.py` | Four figures of the raw data, plus a leakage measurement |
| `brightness_baseline.py` | A one-feature classifier, used to audit the result |
| `evaluate_calibration.py` | ECE, reliability, reject-option sweep, temperature scaling |
| `inspect_patient.py` | Per-slice predictions for one patient, used for the error autopsy |
| `run_smoke.sh` | SLURM smoke test, two epochs, run before any full job |
| `run_train.sh` | SLURM training run. Takes model, seed, epochs and augmentation |
| `run_figures.sh` | SLURM job that rebuilds every figure from a pair of checkpoints |

---

## Dependencies and reproducibility

I measured these on the Rangpur HPC cluster.

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.14.0+cu130 |
| torchvision | matching cu130 build |
| CUDA | 13.0 |
| numpy | used in `train.py`, `predict.py` and the analysis scripts only |
| matplotlib | for all figures |
| Pillow | image loading |

Hardware: one NVIDIA A100-PCIE-40GB.

**Note on `modules.py`.** The assignment says the model code must depend only on
PyTorch. `modules.py` imports nothing else. NumPy appears in `train.py`,
`predict.py` and the analysis scripts, where the assignment allows it.

### What is fixed, and what is not

`set_all_seeds` fixes the seed for Python `random`, for NumPy and for torch.
It fixes the CUDA seed as well. The train and validation split uses the same
seed, so the same patients go to the same split on every run.

An honest limit: `cudnn.benchmark` stays on. It picks the fastest convolution
algorithm for the hardware, and that choice is not guaranteed to be the same
between runs. So the results repeat to within a small margin. They are not
bit-exact. `deterministic=True` would fix that, at a clear cost in speed.

---

## How to run

```bash
# Inspect the dataset
python audit_adni.py

# Produce the data figures
python visualise_data.py --data-root /home/groups/comp3710/ADNI/AD_NC --output-dir figures

# Check the models before you spend GPU time
python modules.py

# Train (SLURM). Arguments: model, seed, epochs, augmentation
sbatch run_train.sh baseline 42 80 strong
sbatch run_train.sh gfnet 42 80 strong

# Run a trained model and draw the figures
python predict.py --model gfnet --checkpoint outputs_gfnet_seed42_strong/best_gfnet.pt --output-dir figures

# Produce the validation predictions that temperature scaling needs
python predict.py --model gfnet --checkpoint outputs_gfnet_seed42_strong/best_gfnet.pt --split validation --no-figures --output-dir figures

# The dilemma investigation
python evaluate_calibration.py \
  --predictions figures/test_predictions_baseline.npz figures/test_predictions_gfnet.npz \
  --val-predictions figures/validation_predictions_baseline.npz figures/validation_predictions_gfnet.npz \
  --names baseline gfnet --output-dir figures
```

`predict.py` and `train.py` need a GPU. Run them through SLURM. Do not run them
on the login node.

---

## The data

```
/home/groups/comp3710/ADNI/AD_NC/
    train/AD   10,400 images from   520 patients
    train/NC   11,120 images from   556 patients
    test/AD     4,460 images from   223 patients
    test/NC     4,540 images from   227 patients
```

File names use the form `<patientID>_<slice>.jpeg`, for example
`1003730_100.jpeg`. Every patient gives exactly 20 slices. Images are 256 × 240,
greyscale, with pixel values from 0 to 255.

Teaching staff confirmed that the number before the underscore is a **patient**
ID, and not a scan ID or a session ID. That matters, because ADNI follows people
over many years and can scan the same person more than once.

![Example slices from different patients](figures/fig1_class_comparison.png)

### The splits, and why they are built this way

Each patient gives 20 slices cut from one 3D scan, so those 20 images look very
similar to each other.

![All 20 slices from one patient](figures/fig2_one_patient_all_slices.png)

A random split at image level would spread the slices of one person across train
and validation. The model could then score well because it recognises the
patient and not the disease. Validation accuracy would look excellent for the
wrong reason, and the test score would then collapse with no clear cause.

So every split happens at **patient** level. I choose which patients go where
first, and all 20 slices of a patient follow that choice.
`check_no_patient_overlap` raises an `AssertionError` if a patient appears in two
splits. It crashes and does not warn, because if that check fires, every number
after it is worthless.

| Split | Images | Patients | NC | AD |
|-------|--------|----------|-----|-----|
| train | 18,300 | 915 | 9,460 | 8,840 |
| validation | 3,220 | 161 | 1,660 | 1,560 |
| test | 9,000 | 450 | 4,540 | 4,460 |

The test split comes exactly as shipped, and I never touched it during
development. The audit confirmed that it shares no patients with the train
folder.

Validation holds 15.0% of the train patients, stratified by class: 78 of 520 AD
patients and 83 of 556 NC patients. Both are within a tenth of a percent of 15%.

Class balance is NC 51.7% and AD 48.3%. That is mild enough that weighted loss
or resampling is not needed. I measured it and did not assume it.

### Preprocessing

**Resize 256 × 240 to 224 × 224.** GFNet needs a fixed square input, because it
works on a grid of patches. I resize and do not crop. A crop could remove the
outer cortex, and that is the tissue that atrophies in Alzheimer's. The stretch
is the same for every image, so the model sees a consistent world.

**Normalisation.** I measured these values over 2,000 train images. I used the
train split only, so nothing leaks in from validation or test.

```
PIXEL_MEAN = 0.1159
PIXEL_STD  = 0.2236
```

The mean is low because these slices are mostly black background. With
placeholder values of 0.5 the normalised range was −1.000 to 0.953, which
squeezed the brain tissue into a narrow band. With the measured values the range
is −0.518 to 3.849. Background sits near −0.52 and tissue spreads across the
positive range.

**Augmentation.** There are two settings. Select one with `--augmentation`.

| Transform | normal | strong |
|---|---|---|
| Horizontal flip | p = 0.5 | p = 0.5 |
| Rotation | ±5° | ±10° |
| Translation | ±5% | ±10% |
| Scale | none | 0.9 to 1.1 |
| Brightness and contrast jitter | none | ±25% |

The brightness jitter is not arbitrary. The brightness experiment below shows
that average intensity carries real signal, which means the model can lean on
brightness as a shortcut. Random brightness during training removes that
shortcut.

An honest caution about the horizontal flip. Alzheimer's atrophy is not always
symmetric between the two hemispheres, so a mirror image does lose a little real
signal. I kept the flip because the extra variety helps more at this dataset
size. But that choice deserves an ablation and not an assumption.

---

## Results

All numbers come from the held-out test split: 9,000 slices from 450 patients.
No model saw them during training or model selection.

### Headline comparison

Both models trained for 80 epochs, seed 42, learning rate 3e-4, batch size 64,
strong augmentation, weight decay 1e-3. The splits are identical.

| Measure | BaselineCNN | GFNet |
|---------|-------------|-------|
| Test accuracy (slice) | **0.7421** | 0.7122 |
| Test accuracy (patient) | **0.7511** | 0.7444 |
| Macro F1 | **0.7357** | 0.7069 |
| AD precision | **0.8411** | 0.7815 |
| AD recall | **0.5913** | 0.5821 |
| NC precision | 0.6892 | 0.6717 |
| NC recall | **0.8903** | 0.8401 |
| Best validation accuracy | 0.9028 | 0.7891 |
| Best epoch (of 80) | 79 | 61 |

The baseline wins on slice accuracy by 3.0 points. At patient level the gap
falls to 0.7 points. GFNet converged at epoch 61. The baseline was still better
at epoch 79.

### Resource profile

| Measure | BaselineCNN | GFNet | Ratio |
|---------|-------------|-------|-------|
| Trainable parameters | 421,474 | 4,788,482 | 11.4× |
| Peak GPU memory | 2.32 GB | 1.58 GB | 0.68× |
| Inference rate | 12,415 img/s | 5,672 img/s | 0.46× |
| Inference latency | 0.081 ms/img | 0.176 ms/img | 2.2× |
| Seconds per epoch | 13.6 | 14.1 | 1.03× |
| Total training time | 18.1 min | 19.0 min | 1.05× |

**GFNet uses less memory, and it has 11.4 times more parameters.** Parameters
and activations are separate costs. The first convolution of the baseline holds
a 224 × 224 × 32 feature map, which is about 1.6 million values per image.
GFNet reduces the image to 196 tokens of 256 values at once, which is about 50
thousand values. The baseline carries large spatial maps through its early
layers. GFNet carries a small token grid all the way through.

![Example predictions](figures/predictions_baseline.png)

---

## Investigation of the open research dilemma

### Does the model learn anatomy, or brightness?

The intensity histograms show that AD tissue is darker than NC tissue on
average. That raised a question I had to answer before I trusted any result.
Does the model read anatomy, or does it read brightness?

![Pixel intensity by class](figures/fig4_intensity_histograms.png)

So I built the simplest classifier I could. `brightness_baseline.py` uses one
number per slice, which is the mean intensity of the tissue pixels. It fits the
threshold on the train split only.

| Classifier | Test accuracy | What it uses |
|---|---|---|
| Chance | 0.500 | nothing |
| Mean tissue brightness | 0.585 | one number per slice |
| GFNet | 0.7122 | learned frequency filters |
| BaselineCNN | 0.7421 | learned convolution features |

The CNN adds **15.7 points** over the brightness rule, so the network learns
real structure. But 0.585 is above chance, so about 8.5 points of the signal is
average brightness. An honest account of this result needs both numbers.

This measurement is also the reason for the brightness jitter in the strong
augmentation setting.

### Does the confidence mean anything?

This is the core of the dilemma. I measured it with `evaluate_calibration.py`.

| Measure | BaselineCNN | GFNet |
|---|---|---|
| Expected Calibration Error | 0.1812 | **0.0934** |
| Worst single bin gap | 0.2603 | **0.1594** |
| Mean confidence when correct | 0.9382 | 0.8281 |
| Mean confidence when wrong | 0.8803 | 0.7499 |
| Separation | 0.0579 | **0.0782** |

![Reliability diagram](figures/calibration_reliability.png)

**GFNet is almost twice as well calibrated as the baseline.** Its calibration
error is 0.0934 against 0.1812, and its worst single bin is out by 0.159 and not
by 0.260. GFNet is also less overconfident in absolute terms. When GFNet is
wrong it claims 75% certainty, not 88%.

This is the clearest result in the report, and it goes against the accuracy
comparison. The baseline predicts better. GFNet has the better idea of what it
does not know.

**Even so, you cannot automate on either confidence score.** The separation
between right and wrong predictions is 0.078 for GFNet. That is 35% better than
the 0.058 of the baseline. It is still far too small for a threshold to split
the two groups cleanly.

![Confidence when right against when wrong](figures/calibration_confidence.png)

### Temperature scaling

Temperature scaling divides the logits by one learned number before the softmax.
It changes no prediction, so accuracy stays the same. But it can make the
confidence honest.

I fit the temperature on the **validation** split. A fit on test, followed by a
report of test calibration, would be a tune on the test set. That is the same
mistake that the patient-level split exists to avoid.

| Model | Temperature | ECE before | ECE after | Change | Accuracy |
|---|---|---|---|---|---|
| BaselineCNN | 1.20 | 0.1812 | 0.1651 | −8.9% | unchanged |
| GFNet | 1.11 | 0.0934 | 0.0787 | −15.7% | unchanged |

A temperature above 1 means the model was overconfident. Both models were, and
both only mildly. Accuracy did not move for either model, which is the expected
behaviour and a check that the implementation is correct.

GFNet gains more from the fit than the baseline does, at 15.7% against 8.9%. So
some of the overconfidence of the baseline is structural, and a single global
scale does not remove it.

**Strong augmentation did most of the work that temperature scaling would have
done.** I ran the same fit on the normal-augmentation models earlier in the
project. The numbers were very different.

| Run | Temperature | ECE before | ECE after | Change |
|---|---|---|---|---|
| GFNet, normal aug | 1.90 | 0.1819 | 0.1102 | −39% |
| GFNet, strong aug | 1.11 | 0.0934 | 0.0787 | −16% |
| BaselineCNN, normal aug | 1.36 | 0.2164 | 0.1963 | −9% |
| BaselineCNN, strong aug | 1.20 | 0.1812 | 0.1651 | −9% |

The fitted temperature for GFNet fell from 1.90 to 1.11. A temperature near 1
means there is little global overconfidence left to remove. Augmentation and
temperature scaling treat the same problem. Augmentation treats it better,
because it also raises accuracy, and temperature scaling cannot.

So the order matters. Fix the training first, then scale the temperature on
what is left.

### Patient-level evaluation

A clinician classifies a patient and not a slice. Each patient gives 20 slices,
so the mean of those 20 probabilities is the clinically useful read-out.

| Model | Slice | Patient | Gain |
|---|---|---|---|
| BaselineCNN | 0.7421 | **0.7511** | +0.9 pts |
| GFNet | 0.7122 | 0.7444 | **+3.2 pts** |

GFNet gains more than three times as much from the mean. That tells us something
about the shape of the errors of each model. You can see the shape when you
count how each patient fails.

| Model | All 20 slices correct | All 20 slices wrong |
|---|---|---|
| BaselineCNN | 226 | 40 |
| GFNet | 157 | **28** |

The mean only helps a patient whose slices disagree with each other. In the
all-wrong group the model has a consistent and confident misread of the anatomy
of that person. No mean fixes that.

**GFNet gets all 20 slices wrong for 30% fewer patients: 28 against 40.** A
patient who is wrong on all 20 slices is a patient the system gets wrong every
time. There is no internal disagreement to flag the case for review. That is a
real safety property, and GFNet holds it with a lower slice accuracy.

The 3.0-point gap in slice accuracy falls to 0.7 points once you evaluate the
way a clinic would.

### The reject option

The rule is simple. If confidence is at or above a threshold, accept the answer
automatically. Below the threshold, send the case to a specialist.

![Reject option curves](figures/reject_option_curves.png)

**Acceptance criterion 4 is not achievable.** No threshold reaches 0.95 accuracy
on the automatic cases and refers 30% of cases or fewer. That follows from the
confidence separation of 0.058. The two groups overlap too much.

The three outcomes do not cost the same. That is why overall accuracy is the
wrong thing to optimise.

| Outcome | Cost |
|---|---|
| False positive, referred | A few minutes of specialist time |
| False positive, auto-flagged | Unnecessary follow-up, patient distress |
| **False negative, auto-cleared** | **A patient goes home with no diagnosis. Nobody checks again.** |

### The decision threshold should move

Both models prefer to say "normal". A move of the AD decision threshold below
0.5 trades NC recall for AD recall.

I measured this on the baseline.

| AD threshold | AD recall | AD precision | NC recall | Accuracy | AD cases missed |
|---|---|---|---|---|---|
| **0.10** | **0.6899** | 0.7598 | 0.7857 | **0.7382** | **1,383** |
| 0.20 | 0.6370 | 0.7909 | 0.8346 | 0.7367 | 1,619 |
| 0.30 | 0.6049 | 0.8149 | 0.8650 | 0.7361 | 1,762 |
| 0.40 | 0.5744 | 0.8315 | 0.8857 | 0.7314 | 1,898 |
| 0.50 (default) | 0.5390 | 0.8492 | 0.9059 | 0.7241 | 2,056 |
| 0.60 | 0.5074 | 0.8628 | 0.9207 | 0.7159 | 2,197 |

At a threshold of 0.10 the model misses 1,383 AD slices and not 2,056. That is
**673 fewer missed cases**, and accuracy goes slightly up and not down.

The default of 0.5 is the wrong operating point for a screening tool.

Note what this does not do. Accuracy stays near 0.73 at every threshold. The
threshold moves errors between the classes. It does not add power to separate
them.

### Overfit: the diagnosis and the fix

The gap from validation to test got wider as training went on. That is the mark
of a model that fits its training group and does not learn something general.

| Baseline run | Validation | Test | Gap |
|---|---|---|---|
| 30 epochs, normal aug | 0.8894 | 0.7241 | 0.165 |
| 80 epochs, normal aug | 0.9326 | 0.7360 | **0.197** |
| 80 epochs, strong aug, weight decay 1e-3 | 0.9028 | 0.7421 | **0.161** |

First I checked whether the two shipped folders differ. The intensity histograms
for the train and test splits look the same, so a shift in the distribution is
not the cause.

The fix was stronger augmentation plus ten times the weight decay. Every measure
moved the right way.

| Measure | Normal aug | Strong aug | Change |
|---|---|---|---|
| Validation | 0.9326 | 0.9028 | −3.0 pts |
| Test (slice) | 0.7360 | 0.7421 | +0.6 pts |
| Test (patient) | 0.7422 | 0.7511 | +0.9 pts |
| Gap | 0.197 | 0.161 | −3.6 pts |
| AD recall | 0.5466 | 0.5913 | **+4.5 pts** |
| Macro F1 | 0.7257 | 0.7357 | +1.0 pts |
| ECE | 0.2164 | 0.1812 | −16% |
| Confidence separation | 0.0408 | 0.0579 | **+42%** |

Validation fell and test rose. That is what less overfit looks like. AD recall
gained the most, and that is the measure that matters in a clinic. It means 199
fewer missed Alzheimer's cases.

---

## Error autopsy

`predict.py` selects the mistakes that the model was **most confident** about,
because those are the dangerous ones. A mistake at 51% confidence goes to a
human under any sensible threshold. A mistake at 99% confidence does not.

![Most confident failures](figures/failures_baseline.png)

> **[FILL IN. This section needs your own observations.]**
>
> There are five false negatives and five false positives in the figure. For
> each one, write two or three sentences. Cover these points:
>
> - the patient ID and the confidence of the model, which `predict.py` prints
> - what the slice looks like: ventricle size, cortex thickness, overall
>   brightness, head position, any artifact
> - your best explanation of what misled the model
>
> Then group the cases. Look for a pattern across them, and do not treat each
> one on its own. Patterns to check for:
>
> - slices near the top or the bottom of the stack, where little brain tissue
>   is visible
> - scans that are unusually bright or unusually dark, which the brightness
>   result predicts
> - heads that are rotated or off-centre in the frame
> - AD patients whose ventricles look normal in size
>
> Close with one sentence. Name the kinds of case you would **not** trust this
> prototype to handle.

---

## Engineering recommendation

**Do not deploy either model as a classifier that works alone.** Neither model
reaches the 0.8 accuracy target. More important, neither model gives a
confidence score you can trust. Acceptance criterion 4 is not achievable with
either model.

**If you must deploy a prototype, use this configuration.**

1. **Decide at patient level and not at slice level.** Take the mean of the 20
   slice probabilities. The patient is the clinically useful unit, and the mean
   is worth about one accuracy point.
2. **Set the AD decision threshold near 0.10 to 0.20, and not 0.50.** This
   saves 673 missed AD cases at no cost in overall accuracy. A missed case is
   the expensive error in screening.
3. **Apply temperature scaling, fitted on validation.** It costs one parameter,
   it changes no prediction, and it cuts the calibration error by 9% to 16%.
   Apply strong augmentation first. Augmentation removes most of the
   overconfidence that temperature scaling would otherwise remove.
4. **Treat every output as a draft for a specialist.** A confidence separation
   of 0.058 means the system cannot find its own unsure cases reliably.

**Does GFNet earn its cost?** The evidence points both ways, so here it is in
full.

| GFNet is worse at | GFNet is better at |
|---|---|
| Slice accuracy, by 3.0 points | Calibration error, 0.0934 against 0.1812 |
| Patient accuracy, by 0.7 points | Confidence separation, 0.078 against 0.058 |
| Parameter count, 11.4× more | Patients wrong on all 20 slices, 28 against 40 |
| Inference latency, 2.2× slower | Peak memory, 0.68× |

The honest summary is that **the baseline is the better classifier and GFNet is
the more trustworthy one**.

A triage system exists to send unsure cases to a specialist. For that job, trust
matters more than accuracy. The calibration error of GFNet is about half the
error of the baseline. GFNet gets all 20 slices wrong for 30% fewer patients.
Its patient-level accuracy is within one point of the baseline. On that reading
GFNet is the better choice, and it costs 11.4 times the parameters.

But that conclusion holds only if the 0.8 target is relaxed. At the required
accuracy neither model qualifies. Then the recommendation is to improve the
input first. Use 3D volumes, or stacks of several slices, before any choice
between these two architectures.

---

## Limitations

**The project did not meet the 0.8 accuracy target.** The best result is 0.7421
at slice level and 0.7511 at patient level. I tried three things and none of
them closed the gap.

| What I tried | Gain |
|---|---|
| 80 epochs instead of 30 | 1.2 points |
| The patient-level mean | 0.9 points |
| Stronger regularisation | 0.6 points |

My view on the cause is that one 2D slice may not hold enough information. The
ADNI literature mostly uses 3D volumes, or several slices, as input. The models
here also did not fully converge. The best validation epoch was 79 of 80, so a
longer schedule would probably give a little more.

**`cudnn.benchmark` is on.** So the results repeat to within a small margin, and
they are not bit-exact.

**One seed.** Every result uses seed 42. A difference of one or two points
between two models could be run-to-run variance. Three seeds per model, with a
mean and a standard deviation, would make the comparison firmer. Time was the
constraint here, and not a choice of method. This is the first thing I would
add.

**The horizontal flip** assumes a left-right symmetry that Alzheimer's atrophy
does not always keep. It deserves an ablation.

---

## Artificial Intelligence Usage Disclosure

> **[READ THIS SECTION AND EDIT IT UNTIL IT IS ACCURATE. It must describe what
> you did, in your words. Do not submit it unchecked.]**

**Tool used.** Anthropic Claude, in an agentic coding session, over about one
week.

**What I used it for.**

- First versions of `modules.py`, `dataset.py`, `train.py`, `predict.py`,
  `evaluate_calibration.py`, `visualise_data.py` and `brightness_baseline.py`.
  I then read, edited and corrected each file.
- A draft of the structure and the text of this README, which I then edited.
- Help with SLURM configuration and with Python errors.
- Ideas for analyses: the brightness ablation, temperature scaling, the
  patient-level mean, and the overfit fix.

**How I checked the output.**

- `modules.py` holds six self-tests. I ran them before any GPU time. The most
  important test sets every GFNet filter weight to 1 + 0i. It then confirms that
  the FFT round trip returns its input without change, to within 4.8e-07.
- `dataset.py` holds an assertion that raises an error if a patient appears in
  two splits. I ran it and checked the reported split sizes by hand. 915 + 161
  equals 1,076 patients, which matches the audit.
- I checked the classification metrics by hand against the confusion matrix. For
  the baseline, 4,042 + 2,637 correct out of 9,000 gives 0.7421. That matches
  the reported accuracy.
- Teaching staff confirmed that the filename IDs are patient IDs.
- Several AI suggestions were wrong and I found them. One was an incorrect SLURM
  partition that came from stale notes. Three separate edits to `train.py` gave
  a duplicate keyword argument, a stray bracket, and a lost `.to(device)` call.
  Each one failed a job before I found it.

**What I did myself.** I chose the project and the model. I decided to split by
patient. I did the reading on Alzheimer's anatomy that informs the error
autopsy. I interpreted every result, and I wrote the engineering recommendation.

I am responsible for every line of code and every claim in this report.

---

## References

[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global filter networks for image
classification," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, vol. 45, no. 6, pp. 7837–7850, 2023.

[2] Alzheimer's Disease Neuroimaging Initiative. http://adni.loni.usc.edu

[3] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern
neural networks," in *Proc. ICML*, 2017. This paper is the source of the
temperature scaling method and of the Expected Calibration Error metric used in
this report.
