# GFNet for Alzheimer's Classification on ADNI Brain MRI

**Manav Sachdeva (s4967283).** COMP3710 Pattern Analysis Project.
Project 2.1 Classification. Model: GFNet. Difficulty: Hard.

---

## Contents

This page carries the problem, the method, the results and the recommendation.
The detail sits in seven linked pages, so you can go straight to the part you
want.

| Page | What is in it |
|---|---|
| [Feasibility review](docs/feasibility-review.md) | The one-page review from the midpoint check-off, with its acceptance criteria and outcomes |
| [Running the code](docs/running-the-code.md) | Repository layout, dependencies, and every command with its real expected output |
| [The data and the splits](docs/data-and-splits.md) | Dataset layout, why splits are patient-level, and the preprocessing |
| [Dilemma investigation](docs/dilemma-investigation.md) | Brightness ablation, calibration, temperature scaling, patient-level evaluation, reject option, overfitting |
| [Error autopsy](docs/error-autopsy.md) | Four named failure patients, what the models did to them, and why |
| [The scan-ID finding](docs/scan-id-finding.md) | Evidence that the filename ID identifies a scan and not a person |
| [Experiment ledger](docs/experiment-ledger.md) | All 17 experiments, including the two that disproved my own hypotheses |

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

---

## Results

All numbers come from the held-out test split: 9,000 slices from 450 patients.
No model saw them during training or model selection.

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

![BaselineCNN example predictions](figures/predictions_baseline.png)

![GFNet example predictions](figures/predictions_gfnet.png)

---

---

## The three findings that matter

### 1. GFNet is twice as well calibrated, and loses on accuracy

| Measure | BaselineCNN | GFNet |
|---|---|---|
| Expected Calibration Error | 0.1812 | **0.0934** |
| Confidence separation | 0.0579 | **0.0782** |
| Patients wrong on all 20 slices | 40 | **28** |

The baseline predicts better. GFNet has the better idea of what it does not
know. Neither confidence score is good enough to automate on, so acceptance
criterion 4 is reported as not achievable.

[Read the full investigation &rarr;](docs/dilemma-investigation.md)

### 2. The filename ID identifies a scan, not a person

Near-duplicate patients sit at a median ID gap of 2. Ordinary pairs sit at
47,026. So 16.1% of validation patients share a person with the training set,
even though the patient-level split and its overlap assertion are both correct.

Measured cost: 0.6 points of validation accuracy. **These splits are leak-free
by ID and are not leak-free by subject.**

[Read the evidence &rarr;](docs/scan-id-finding.md)

### 3. The constraint is the cohort, not the model

Five GFNet configurations across four separate axes: training length,
regularisation, token resolution, and input representation.

**Validation moved 5.3 points. Test moved 3.2, and not in step with it.** Not
one run crossed 0.72.

The best GFNet result is 0.7161 at slice level, 8.4 points short of the 0.80
minimum the specification sets. The campaign is complete rather than abandoned,
and the ledger records all 17 experiments.

[Read the ledger &rarr;](docs/experiment-ledger.md)

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
   the expensive error in screening. The error autopsy makes this urgent: all
   twenty of the worst total failures, across both models, are patients who
   have AD.
3. **Apply temperature scaling, fitted on validation.** It costs one parameter,
   it changes no prediction, and it cuts the calibration error by 9% to 16%.
   Apply strong augmentation first. Augmentation removes most of the
   overconfidence that temperature scaling would otherwise remove.
4. **Treat every output as a draft for a specialist.** A confidence separation
   of 0.058 means the system cannot find its own unsure cases reliably.
5. **Run both models and refer any patient they disagree about.** Only 20 of
   the 450 test patients defeat both architectures. The baseline fails
   completely on 20 patients that GFNet partly reads, and GFNet on 8 that the
   baseline partly reads. Disagreement between a convolutional model and a
   frequency-domain model is a cheap and useful signal. It is also independent
   of the confidence score, which the autopsy shows cannot be trusted alone.

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

**The splits are leak-free by ID and not by subject.** 16.1% of validation
patients share a person with the training set, because the filename ID
identifies a scan rather than a person. Measured cost: 0.6 points of validation
accuracy and 0.2 points of test accuracy. The section above sets out the
evidence and the fix I would apply with more time.

**The validation-to-test gap of 16 points is unexplained.** Folder statistics
and subject leakage were both tested and neither accounts for it. I am
recording it as open rather than attributing it to a cause I cannot
demonstrate.

---

---

## Artificial Intelligence Usage Disclosure

# Tool used:
Anthropic Claude, mainly through an agentic coding session over approximately one week.

# What I used it for:
Claude helped me create the initial versions of several project files, including modules.py, dataset.py, train.py, predict.py, evaluate_calibration.py, visualise_data.py, and brightness_baseline.py. These were used as starting points rather than final solutions. I reviewed each file, made edits, corrected mistakes, and adapted the code to fit the requirements of the project.
It also helped generate an early draft of the README structure and some explanatory text, which I substantially revised and rewrote during the final documentation process.
During development, I used Claude as a debugging assistant when dealing with Python errors and configuring SLURM jobs on the university cluster. It was also useful for brainstorming potential analyses, including the brightness-ablation experiment, temperature scaling for calibration, patient-level prediction averaging, and strategies for addressing overfitting.

# How I checked the output:
I did not accept Claude's suggestions without verification. The project includes several checks that allowed me to validate the code before running expensive training jobs.
In modules.py, I implemented and ran six self-tests. The most important test sets every GFNet filter weight to 1+0iand verifies that the FFT round-trip reconstruction reproduces the original input to within 4.8×〖10〗^(-7). This confirmed that the frequency-domain operations were functioning correctly.
In dataset.py, I used an assertion to ensure that no patient appeared in more than one data split. I also manually checked the reported split sizes. The training and validation splits contained 915 and 161 patients respectively, giving a total of 1,076 patients, which matched the dataset audit.
I independently verified the evaluation metrics by calculating them directly from the confusion matrix. For example, in the baseline model, 4042+2637=6679correct predictions out of 9000 samples, giving an accuracy of 0.7421, which matches the reported result.
I also confirmed with teaching staff that the identifiers contained in the filenames correspond to patient IDs, which was important for ensuring that splitting by patient was implemented correctly.
Several Claude-generated suggestions were incorrect and had to be fixed. One example was an invalid SLURM partition recommendation that appeared to come from outdated information. Other mistakes introduced during code edits included a duplicated keyword argument, a stray bracket causing a syntax error, and a missing .to(device) call. Each issue caused failed runs that I then diagnosed and corrected myself.

# What I did myself:
I selected the project topic, chose GFNet as the model architecture, and decided to perform the dataset split at the patient level rather than the image level. I carried out the background reading on Alzheimer's disease anatomy that informed the error analysis and interpretation of the results.
All experimental decisions, result interpretation, discussion, conclusions, and recommendations are my own work. While Claude assisted with drafting code, troubleshooting, and generating ideas, I was responsible for verifying every implementation, testing every change, interpreting every result, and deciding what was ultimately included in the final report.
I take responsibility for all code, analyses, and claims presented in this submission.


## References

[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global filter networks for image
classification," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, vol. 45, no. 6, pp. 7837–7850, 2023.

[2] Alzheimer's Disease Neuroimaging Initiative. http://adni.loni.usc.edu

[3] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern
neural networks," in *Proc. ICML*, 2017. This paper is the source of the
temperature scaling method and of the Expected Calibration Error metric used in
this report.
