## Summary

GFNet (Global Filter Network) for Alzheimer's classification on the ADNI brain
MRI dataset. It is benchmarked against a convolutional baseline trained on
identical splits. Hard difficulty tier, Project 2.1 Classification.

The open engineering dilemma for this domain is confidence, not accuracy. Does
the confidence of a model tell you whether it is right? Can a confidence
threshold safely automate the easy cases? This submission answers both with
measurements.

All code is in `recognition/GFNet_s4967283/`. The report is the README in that
folder.

## Headline results

Held-out test split, 9,000 slices from 450 patients, seed 42.

| Measure | BaselineCNN | GFNet |
|---|---|---|
| Accuracy, slice level | **0.7421** | 0.7161 |
| Accuracy, patient level | **0.7511** | 0.7444 |
| Expected Calibration Error | 0.1812 | **0.0934** |
| Confidence separation | 0.0579 | **0.0782** |
| Patients wrong on all 20 slices | 40 | **28** |
| Trainable parameters | **421,474** | 4,788,482 |
| Peak GPU memory | 2.32 GB | **1.58 GB** |

The baseline is the better classifier. GFNet is the better calibrated one, by
about a factor of two. The README works through the trade-off. It recommends
against the use of either model as a classifier that works alone.

## On the 0.80 accuracy minimum

The specification sets a minimum of 0.80 test accuracy for GFNet. **It was not
reached.** The best result is 0.7161 at slice level and 0.7444 at patient
level.

The README holds a section called "Experiments towards the 0.80 minimum". It
records every experiment run towards that number. Fifteen entries, and two of
them are hypotheses that the measurements disproved.

The short version: four training configurations and one architectural change
were tried. Validation accuracy moved 5.3 points across them. Test accuracy
moved 3.2 points, and not in step with validation.

Learning rate, epoch count, augmentation strength, weight decay, token
resolution and parameter count all change one thing. They change how well the
model fits the training cohort. None of that transfers to the test cohort.

So the constraint is the difference between the two cohorts, not the model.

## Investigation of the dilemma

- **Brightness ablation.** A classifier that uses only mean tissue intensity
  reaches 0.585. The CNN adds 15.7 points over it, so the network learns real
  structure. But about 8.5 points of the signal is average brightness. This
  measurement is the reason for the brightness jitter in the strong
  augmentation setting.
- **Calibration.** The ECE of GFNet is about half that of the baseline.
  Neither model separates right from wrong well enough to automate on.
  Acceptance criterion 4 is reported as not achievable rather than dropped
  quietly.
- **Temperature scaling**, fitted on validation only. Strong augmentation
  lowered the fitted temperature of GFNet from 1.90 to 1.11. Augmentation and
  temperature scaling treat the same overconfidence. This corrected a claim
  made earlier in the project.
- **Decision threshold.** At 0.10 rather than 0.50, the baseline misses 673
  fewer AD cases at no cost in overall accuracy.
- **Error autopsy.** The baseline returns P(AD) below 0.00005 on all 20 slices
  of a patient who has AD. All twenty of the worst total failures across both
  models are missed AD cases. The catastrophic failures run entirely in the
  clinically expensive direction.

## A finding about the dataset itself

The filename ID identifies a **scan, not a person**.

Near-duplicate patients sit at a median ID gap of 2. Ordinary pairs sit at
47,026. All 26 leaked validation patients are within 20 IDs of their match.
None of the 446 clean test patients are. The IDs run in sequence, so the repeat
scans of one person land next to each other.

So 16.1% of validation patients share a person with the training set. The
patient-level split and its overlap assertion are both correct, and neither
catches this. Measured cost: 0.6 points of validation accuracy and 0.2 points
of test accuracy.

**These splits are leak-free by ID and are not leak-free by subject.** The
README says so and quantifies it. It also sets out the subject-level grouping
that would fix it. Teaching staff may want to know about this on its own
account, separately from this submission.

## Methodology notes for the reviewer

- **Splits are at patient-ID level**, with an assertion that raises rather than
  warns. See the caveat above.
- **The test split is used exactly as shipped.** It was never touched during
  development. Normalisation constants and every threshold are fitted on the
  training split only. The temperature is fitted on validation only.
- **The ensemble weight is not tuned on test.** The sweep is reported and the
  tuned value is explicitly not claimed. The report uses the even split, which
  needs no fitting.
- **Known limitations**, all stated in the README:
  - the 0.80 target was not met
  - the validation-to-test gap of 16 points is unexplained and recorded as open
  - `cudnn.benchmark` is on, so results repeat to a small margin and are not
    bit-exact
  - every result uses a single seed

## How to run

```bash
python modules.py                              # self-tests, no GPU needed
sbatch run_train.sh gfnet 42 80 strong         # train
sbatch run_figures.sh                          # rebuild every figure
```

Dependencies, exact versions and the full command list are in the README.

## Attribution

No code blocks were reused from online sources.

GFNet follows the architecture described in Rao et al., *Global Filter Networks
for Image Classification*, IEEE TPAMI 45(6), 2023. The implementation here is
my own. Temperature scaling and the Expected Calibration Error metric follow
Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. Both papers
are cited in the README.

Generative AI was used during this project. The mandatory Artificial
Intelligence Usage Disclosure is a section of the README. It sets out what the
tool was used for, and how I checked the output.
