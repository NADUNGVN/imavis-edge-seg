# PACE-Seg: Hardware-Cost-Conditioned Elastic Semantic Segmentation with Calibrated Risk Routing for Edge Accelerators

Anonymous submission to *Image and Vision Computing*

## Abstract

Semantic segmentation on heterogeneous edge accelerators must balance image-dependent prediction difficulty against platform-dependent latency. A fixed model exposes only one operating point, while an elastic network exposes several candidates but does not determine which candidate should process a particular image under a deployment budget. We present PACE-Seg, an elastic segmentation system with four width-depth-resolution levels and a runtime policy that maps the entropy of one inexpensive probe prediction to a separate expected-error estimate for every candidate. The policy selects the least costly candidate predicted to meet a risk target while enforcing a hard budget using directly measured end-to-end route costs. Calibration is performed separately for each evaluated domain or adverse-condition subset; the experiment therefore assumes that the deployment domain is known or configured and does not evaluate a single condition-agnostic calibrator. We evaluate Cityscapes and the fog, night, rain, and snow subsets of ACDC. A three-run comparison shows that the largest elastic level remains within 1.1 mIoU points of a similarly sized independently trained Fast-SCNN on every split. Across four edge devices, the evaluated levels span a 110.2-fold FLOPs range but only a 10.3--19.5-fold measured-latency range; a simple proportional FLOPs proxy consequently mis-selects candidates and can violate latency budgets. For the final routing evaluation, we measure complete warm-route latency on a TensorRT/CUDA accelerator and a Hailo-8 dataflow accelerator, then replay held-out per-image predictions with those measured costs. Over three separately trained seed-labelled checkpoints and two backends, the proposed policy outperforms or matches a rank-based calibrated-risk policy in 67 of 69 conditionally fair operating cells (51 wins, 16 ties, and 2 losses), with a macro-averaged gain of 0.0241 mIoU. It has no budget-violating operating cell among 120 evaluated cells, compared with 51 for the baseline; this zero count is partly a consequence of the hard constraint by design. The two backend evaluations share segmentation predictions, and the operating cells are correlated, so neither count is treated as an independent-sample statistical result. We additionally show that EMA-percentile activation calibration keeps fake-quantized QAT degradation within 1.5 mIoU points for every tested seed and elasticity level, without claiming compiled INT8 accuracy. The results support hardware-cost-conditioned runtime selection within the evaluated, condition-specific calibration protocol while exposing limitations in uncertainty transfer, rank-dominated budgets, compiled-accuracy validation, and energy measurement.

**Keywords:** semantic segmentation; elastic neural networks; dynamic inference; uncertainty calibration; edge acceleration; hardware-aware deployment

## 1. Introduction

Semantic segmentation is increasingly deployed on edge systems in autonomous driving, robotics, and augmented reality. Such systems are heterogeneous: two accelerators may assign very different latencies to the same network even when its parameter count and floating-point operation count are unchanged. Prediction quality also varies with the input. Clear daytime images and adverse scenes containing fog, night, rain, or snow need not warrant identical computation. Accuracy alone is therefore insufficient for selecting a deployable segmentation model, and an abstract complexity measure is not a substitute for the route cost observed on the target platform.

Conventional deployment selects one architecture for all inputs. Compact real-time segmentation networks provide useful static speed--accuracy trade-offs, but a single operating point can spend unnecessary computation on easy images and cannot exploit additional budget on harder ones. Training several independent networks creates more operating points but multiplies training, validation, packaging, and maintenance activities. The extent of that overhead is not measured in this work, so we do not claim a quantified reduction in lifecycle cost.

Elastic and slimmable networks address part of this problem by sharing parameters among subnetworks of different capacities. Once trained, one model can expose multiple deployment candidates. Elasticity alone, however, does not specify which candidate should process an image, how uncertainty from a cheap candidate transfers to the others, or how a selection rule should obey an accelerator-specific latency budget. These questions are distinct from producing a set of subnetworks.

Dynamic inference addresses input-dependent computation through gates, early exits, or spatially adaptive routes. Hardware-aware neural architecture search addresses a different axis: it uses measured latency or a latency predictor to select an architecture at design time. PACE-Seg studies their intersection. The available candidates are fixed by an elastic segmentation network; runtime selection is image-dependent; and feasibility is defined by directly measured route costs, including the probe, uncertainty computation, policy, engine dispatch or activation, and any additional candidate inference. This is hardware-cost-conditioned runtime routing, not continuous architecture search and not hardware-aware training.

Uncertainty estimation adds another difficulty. A high-entropy probe prediction may indicate a difficult image, but one scalar does not directly state the expected error of every candidate. A rank-step policy also ignores whether adjacent candidates are separated by one millisecond or tens of milliseconds. We therefore fit, separately within each evaluated domain or condition, a candidate-specific monotonic mapping from a shared probe score to each candidate's observed error and enforce a hard measured-cost budget during selection. The resulting evidence concerns deployments in which the appropriate domain-specific calibration set is known; unknown-condition selection is outside the experiment.

The study asks four bounded questions. First, can measured accelerator cost change candidate selection relative to a simple proportional FLOPs proxy? Second, can a shared elastic network approach the quality of a similarly sized independently trained model? Third, does candidate-specific calibrated routing preserve a favorable quality--latency trade-off after its real route overhead is included? Fourth, can a bounded fake-quant QAT procedure preserve the elastic model's accuracy without being misrepresented as compiled INT8 deployment?

The contributions are:

1. We integrate a four-level elastic semantic-segmentation model with per-domain, candidate-specific expected-error calibration from one shared inexpensive probe. Elastic training, sandwich sampling, distillation, and boundary supervision are inherited techniques; the contribution is their use as the candidate family for the measured-budget routing mechanism.
2. We formulate a runtime policy that selects among those candidates under a hard budget using directly measured end-to-end route costs. This differs from design-time hardware-aware NAS and from rank-only escalation.
3. We validate the route-cost accounting on two structurally different accelerators. TensorRT/CUDA requires accelerator-resident entropy computation to avoid a severe host-side bottleneck, whereas Hailo-8 is dominated by mandatory network-group activation and deactivation.
4. We provide a bounded empirical analysis of three supporting issues: failure of a simple proportional FLOPs proxy in the evaluated setting, near-parity between the largest elastic level and a matched Fast-SCNN, and the behavior of dynamic, maximum-observer, and EMA-percentile fake-quant QAT.

The claims are intentionally narrower than universal reliability or robustness. The principal routing comparison uses held-out halves of validation splits, offline per-image replay, shared predictions across backends, and correlated operating cells. These boundaries are part of the result rather than post-hoc qualifications.

## 2. Related work

### 2.1. Efficient and real-time semantic segmentation

Efficient segmentation research seeks to preserve spatial detail and context while reducing latency. BiSeNet separates spatial and contextual processing, STDC removes redundancy from bilateral designs, and PIDNet adds a boundary-oriented branch to control the interaction between detail and context [1--3]. Transformer-based models such as SegFormer provide another family of accuracy--efficiency operating points [4]. Collectively, these methods demonstrate that architecture design can substantially improve the static quality--latency trade-off.

The remaining limitation for the present problem is not a lack of compact architectures. A static network commits to one operating point for every image and every budget. Even when several static networks are available, their runtime selection remains external to the architecture. PACE-Seg therefore treats efficient architectures as comparison points and focuses on exposing and selecting multiple capacities within one elastic model.

### 2.2. Elastic, slimmable, and once-for-all networks

Slimmable neural networks introduced shared weights and switchable normalization for execution at multiple widths, while universally slimmable networks added the sandwich rule and in-place distillation [5,6]. Once-for-All extended the principle across depth, width, kernel size, and resolution and decoupled supernet training from deployment-time specialization [7]. For semantic segmentation, SlimSeg applied channel slimming, downward distillation, and boundary supervision to produce multiple capacities from one model [8]. Multi-Exit Semantic Segmentation Networks pursued a related train-once, customize-at-deployment objective through intermediate exits [9].

These works establish that one training process can expose multiple accuracy--efficiency candidates. They do not, by themselves, solve image-wise selection under a hard budget measured on a specific accelerator. OFA-style specialization is primarily deployment-time subnet selection, SlimSeg exposes adjustable widths, and MESS optimizes exit configurations; PACE-Seg instead asks how a shared probe can estimate the error of each available candidate and how those estimates should interact with complete route costs at runtime.

### 2.3. Dynamic inference and routing for segmentation

Dynamic routing for semantic segmentation learns image-dependent paths through a multi-scale network and can regularize expected computation [10]. Anytime Dense Prediction adds confidence-adaptive exits and spatial masking, while MESS uses early exits to adapt computation to input difficulty [9,11]. These methods show that dense prediction can allocate computation conditionally rather than execute a fixed graph for every input.

PACE-Seg differs in the object being routed and the constraint used. It chooses among separately compiled elastic candidates rather than internal paths or intermediate heads. Its baseline and proposed policies both use one cheap full-image probe, but the proposed policy predicts a distinct error for each candidate and admits only candidates within a measured end-to-end budget. The method is therefore closer to calibrated model selection than to a learned architectural gate. It does not claim end-to-end optimal routing or a formal risk guarantee.

### 2.4. Hardware-aware model selection and NAS for segmentation

Platform-aware NAS established that real latency can be a more faithful objective than FLOPs for mobile deployment [12]. Segmentation-specific systems such as FasterSeg, AutoSegEdge, RealtimeSeg, and edge-oriented latency-predictor search incorporate latency into architecture optimization and deploy searched models on edge hardware [13--16]. These studies collectively establish that deployment cost should enter model design and that FLOPs alone need not preserve latency ordering or magnitude.

The distinction from PACE-Seg is temporal. Hardware-aware NAS makes design-time decisions about an architecture or search space. PACE-Seg does not feed hardware cost back into supernet training and does not perform continuous architecture search. It performs post-training, runtime selection among four fixed candidates. Its RQ1 claim is correspondingly narrow: in the measured four-candidate setting, a simple through-origin proportional FLOPs proxy can reproduce neither the observed cost scale nor all budget-feasible choices. This does not imply that FLOPs-aware NAS in general is inferior, nor that learned latency predictors cannot transfer.

### 2.5. Uncertainty, calibration, and selective segmentation

Confidence calibration for segmentation is affected by prediction correctness, model capacity, crop size, and domain shift [17]. Broader evaluations show that improved in-domain accuracy does not automatically imply calibrated uncertainty under datasets such as ACDC [18]. ACDC itself provides fog, night, rain, and snow scenes and supports uncertainty-aware segmentation through invalid-region annotations and UIoU [19]. Risk--coverage and rejection metrics provide ways to evaluate whether confidence ranks errors, but they do not automatically convert one model's confidence into another model's expected error.

PACE-Seg uses mean softmax entropy from the tiny candidate as a shared feature and fits one monotonic quantile-binned mapping per candidate and per evaluated domain or condition. The approach is calibration-based routing, not a proof that entropy is universally reliable. It also differs from pixel-wise calibration: the output is one image-level expected-error estimate per candidate. This protocol presumes that the appropriate domain-specific calibrator is known at deployment; a pooled or automatically selected unknown-condition calibrator is not evaluated. ACDC/night produces the weakest probe ranking in our evaluation, and ACDC invalid-region labels are not used to report UIoU. These limits prevent a general reliability claim.

### 2.6. Positioning of PACE-Seg

Prior work has separately demonstrated efficient segmentation, elastic subnetworks, input-adaptive computation, hardware-aware architecture selection, and uncertainty-aware prediction. PACE-Seg inherits elastic training and entropy-based confidence estimation. Its contribution lies in combining a shared probe with per-domain candidate-specific error mappings, enforcing a hard budget defined by measured complete route costs, and auditing the resulting policy on TensorRT/CUDA and Hailo-8. The evidence is deployment-oriented and integrative: it does not establish priority for slimmable segmentation, dynamic routing, calibration, or hardware-aware search as individual ideas, nor does it establish condition-agnostic calibration.

## 3. Method

### 3.1. Elastic candidate family

PACE-Seg contains four discrete levels: tiny, small, medium, and large. Their width multipliers are 0.25, 0.50, 0.75, and 1.00; their repeated block counts are 2, 3, 4, and 6; and their input resolutions are 384x192, 512x256, 768x384, and 1024x512 pixels. The levels share convolutional weights through active-channel slicing and use level-specific batch-normalization state. The operator set is restricted to convolutions, normalization, standard activations, pooling, elementwise operations, concatenation, and static-factor bilinear resizing that passed compiler smoke tests in the evaluated toolchains. Because no unconstrained-operator control was trained, compiler compatibility is deployment evidence rather than a measured accuracy contribution.

Training uses the sandwich rule: each step evaluates the smallest and largest levels and one randomly selected intermediate level. The largest level receives supervised segmentation loss, smaller levels receive supervised and in-place distillation terms, and a boundary auxiliary term emphasizes semantic transitions. The default run uses 100,000 optimization steps, batch size 8, AdamW with weight decay 1e-4, initial learning rate 3e-4, cosine decay, 500 warm-up steps, and gradient clipping at 5.0. The data stream combines Cityscapes and ACDC with random scale/crop/pad, horizontal flip, and color jitter.

The four levels are exported as separate static graphs. Runtime routing therefore selects an engine before the selected candidate is executed; no compiled graph contains data-dependent control flow.

### 3.2. Shared probe and candidate-specific error calibration

Let the tiny candidate produce logits z(x). The raw probe score is mean pixel entropy,

\[
s(x)=\frac{1}{|\Omega|}\sum_{u\in\Omega}-\sum_{c=1}^{C}p_{u,c}\log\max(p_{u,c},10^{-12}),
\]

where p is the softmax of z and C=19. The score is image-level and requires no separate probe network.

For each candidate l, we fit a monotonic calibrator g_l that maps the same s(x) to the candidate's observed per-image pixel error. A separate set of candidate calibrators is fitted for Cityscapes and for each ACDC condition; calibrators are not pooled across the five splits. Fit-half samples are divided into ten equal-count bins by probe score. The target in each bin is the mean observed error of candidate l, and a cumulative-maximum pass enforces non-decreasing expected error. Thus

\[
\hat r_l(x)=g_l(s(x))
\]

is candidate-specific calibration from shared sensing, not candidate-specific probing. At deployment, this formulation requires the relevant domain or condition to be known or otherwise configured so that the corresponding calibrator set is selected.

The implementation computes fit-half entropy only over ground-truth-valid pixels but computes inference and held-out entropy over all pixels because an ignore mask is unavailable at deployment. This does not expose held-out ground truth to the policy, but it is a fit--deployment feature mismatch. We retain the frozen evidence and disclose the mismatch rather than describing the calibrator as perfectly deployment-matched.

### 3.3. Hardware-cost-conditioned policy

For a target device, candidate l has measured route cost t_l, risk target tau, and budget B. The proposed policy D forms the feasible set

\[
S_B=\{l:t_l\le B\}.
\]

If at least one feasible candidate satisfies \(\hat r_l(x)\le\tau\), D selects the cheapest such candidate. Otherwise it selects the feasible candidate with minimum predicted error, breaking ties by lower cost. If no candidate is within budget, it selects the globally cheapest candidate. The final fallback can still exceed a budget smaller than the cheapest available route; the evaluated budget grid uses the four measured route costs, so this case does not occur in the reported final cells.

We compare D with three progressive ablations. A uses one calibrator for the probe candidate and escalates by latency rank according to integer multiples of tau. B retains the single risk value but targets a latency magnitude before choosing the closest candidate. C uses candidate-specific risks but has no explicit budget. The progression is not a factorial 2x2 because D adds a hard-budget axis absent from A--C. Static candidates, uncalibrated entropy routing, and a ground-truth oracle are additional references. The oracle is used only as an upper bound and never supplies information to a deployable policy.

### 3.4. Fit, selection, and held-out protocol

Each Cityscapes or ACDC validation split is partitioned by alternating index: even-index images form the fit half and odd-index images form the held-out half. Every candidate is evaluated once per image and its prediction is cached. Within each split, that split's candidate calibrators are fitted only from its fit-half probe scores and candidate errors; there is no cross-condition pooling. Five risk targets are the 10th, 25th, 50th, 75th, and 90th percentiles of fit-half observed error, macro-aggregated across the five splits. One canonical risk grid is generated per training run and reused for both E1 and E3; only the measured cost table changes.

For each budget and policy, the representative risk target is selected using fit-half statistics: among points whose fit-half mean cost does not exceed the budget, choose the highest fit-half mIoU and break ties by lower cost. If no point is feasible, the cheapest point is marked infeasible. Held-out predictions are read only after this choice. The final overhead replay implements this separation for both A and D. An older progressive-evaluation path selected D's point from held-out values, but those outputs are not used for the canonical final claim.

The oracle chooses, independently for each held-out image, the lowest-ground-truth-error candidate within the budget. It bounds available candidate quality; it is not a deployable policy.

### 3.5. Fake-quantized QAT

QAT replaces each convolution with per-tensor symmetric 8-bit fake quantization of its input activation and active weight tensor, using PyTorch's straight-through estimator. Dynamic mode derives the activation scale from every call. Calibrated mode freezes one activation range per layer after a 200-image calibration pass spanning clean and adverse conditions. The original maximum observer retains the largest activation observed over all calls. The final EMA-percentile observer computes the 99.9th percentile of absolute activation in each call and combines successive values with momentum 0.9.

For a slimmable convolution, one calibration buffer belongs to the shared layer and aggregates calls from all elasticity levels; the implementation does not maintain a separate frozen activation scale for each level. Each QAT run fine-tunes 10,000 steps from an FP32 checkpoint at learning rate 3e-5. These are simulated fake-quant results. They neither execute integer kernels nor establish compiled TensorRT or Hailo INT8 accuracy.

## 4. Experimental protocol

### 4.1. Datasets and metrics

Cityscapes supplies 2,975 training and 500 validation images. ACDC supplies 1,600 training and 406 validation images distributed across fog, night, rain, and snow. Both use the 19 Cityscapes train classes. Segmentation quality is dataset-level mean intersection-over-union. Router calibration targets per-image pixel error, while final routed quality is recomputed by accumulating confusion matrices from the selected candidate and calculating mIoU once over the routed set.

The main router unit is an operating cell defined by one training run, backend, dataset split, and budget. Cells share images, predictions, thresholds, and often candidate-feasible sets. We therefore report descriptive counts and macro averages without p-values or confidence intervals that treat cells as independent samples.

### 4.2. Training runs and baselines

The near-parity comparison uses three augmented supernet runs and three augmented Fast-SCNN runs. The final router package uses the three separately trained seed-labelled checkpoints summarized below. Repository naming indicates that two runs are augmentation-labelled and one is the earlier unlabelled configuration; checkpoint hashes, initialization-parent identifiers, and immutable full training manifests are not recorded in the final report. We consequently call them separate training runs rather than a controlled three-seed estimate under a proven identical configuration.

**Training-checkpoint provenance used by the final router package.**

| Run label | Repository experiment ID | Seed label | Checkpoint | Augmentation provenance | Missing immutable provenance |
|---|---|---:|---|---|---|
| run 0 | `pace_seg_v1_aug_seed0` | 0 | step 100,000 | augmentation-labelled ID | checkpoint/config hash; initialization parent |
| run 2 | `pace_seg_v1_seed2` | 2 | step 100,000 | earlier unlabelled ID | checkpoint/config hash; initialization parent |
| run 3 | `pace_seg_v1_aug_seed3` | 3 | step 100,000 | augmentation-labelled ID | checkpoint/config hash; initialization parent |

Fixed-architecture comparisons include Fast-SCNN, BiSeNetV2, SegFormer-B0, DDRNet-23-slim, and MobileNetV3+DeepLabV3. Fast-SCNN is the parameter-matched RQ2 comparator for the large elastic level. Static-small, static-large, raw entropy, rank-based calibrated policy A, latency-spacing policy B, candidate-specific budget-unaware policy C, constrained policy D, and an oracle provide routing references.

### 4.3. Device-cost measurement

Candidate-only latency is measured on four devices: E1, a Raspberry Pi 5 host with a Hailo-8 M.2 accelerator; E2, a Jetson Xavier NX; E3, a Jetson AGX Xavier; and E5, a Jetson Orin Nano. NVIDIA devices use TensorRT FP16. Measurements use batch size one, warm-up, three independent benchmark runs for the candidate lookup table, and bootstrap intervals. Energy is excluded because no external calibrated power meter was available.

The final router claim uses complete warm route costs on E1 and E3. On E3, all four TensorRT contexts are resident. A route measures tiny inference, a GPU-resident entropy kernel, policy computation, context dispatch, and additional selected-candidate inference; tiny output is reused when tiny is selected. On E1, all HEFs are configured but the hardware permits only one active network group, so each inference pays activation and deactivation. Entropy runs on the host because Hailo-8 has no CUDA-like general-purpose kernel path. Each warm route class uses 50 warm-up iterations and at least 500 timed iterations. Cold reload is measured as a stress case but is not used in the final frontier replay because E1 and E3 impose non-equivalent cold semantics.

The route harnesses use compiled graphs with the correct post-rescale architecture but do not execute trained checkpoint weights for the accuracy evaluation. The final result combines measured route latency with cached trained-model predictions in an offline, per-image policy replay. It is not an integrated on-device accuracy run.

### 4.4. Fair-cell and violation definitions

A fair A-versus-D cell is one in which A has zero held-out per-image budget violations at its fit-selected operating point. This conditioning excludes cells where the baseline obtains quality by exceeding its own budget; it does not assert that the remaining 69 cells are random or independent. Because the definition conditions on A's success, it is more favorable to A than pooling its infeasible wins with feasible comparisons. We accompany the fair-cell quality comparison with violation counts over all 120 cells.

An operating cell is called violating when at least one held-out image exceeds the cell's budget. This differs from the mean violation rate within such a cell. D's zero violating-cell count is expected in part because D filters candidates by a hard budget. The empirical question is whether D retains useful quality while satisfying that constraint.

## 5. Results

### 5.1. Elastic quality and same-budget near-parity

The trained levels form a monotonic quality ladder on Cityscapes and every ACDC condition. The headline RQ2 comparison is the large level against an augmented, similarly sized Fast-SCNN.

**Table 1. Three-run mean mIoU for the matched RQ2 comparison. Positive gap favors the elastic large level.**

| Split | Fast-SCNN | PACE-Seg large | Gap |
|---|---:|---:|---:|
| Cityscapes | 0.5228 | 0.5338 | +0.0110 |
| ACDC/fog | 0.5629 | 0.5650 | +0.0021 |
| ACDC/night | 0.3822 | 0.3757 | -0.0065 |
| ACDC/rain | 0.5050 | 0.5027 | -0.0023 |
| ACDC/snow | 0.5048 | 0.5055 | +0.0007 |

The elastic level is higher on three splits and lower on two; every absolute difference is at most 0.0110 mIoU. This supports near-parity, not superiority. It also does not quantify total training or maintenance savings.

### 5.2. FLOPs does not reproduce measured cost in this candidate set

The four levels require 0.2060, 1.1678, 6.1712, and 22.6992 GFLOPs. Their large-to-tiny ratio is 110.2. Measured end-to-end candidate latency grows much less sharply.

**Table 2. Candidate-only mean latency used for the four-device RQ1 analysis.**

| Level | E1 Hailo-8 | E2 Xavier NX | E3 AGX Xavier | E5 Orin Nano |
|---|---:|---:|---:|---:|
| tiny | 3.714 ms | 2.061 ms | 0.912 ms | 1.081 ms |
| small | 6.597 ms | 4.964 ms | 1.781 ms | 2.604 ms |
| medium | 20.29 ms | 15.10 ms | 4.546 ms | 7.727 ms |
| large | 41.14 ms | 40.26 ms | 9.389 ms | 17.95 ms |

Measured large-to-tiny ratios range from 10.3 to 19.5, so the FLOPs ratio overstates cost scaling by roughly 6--11 times. Under a 10 ms budget, measured-cost selection chooses small on E1 and E2, large on E3, and medium on E5. A through-origin proportional FLOPs-to-latency fit calibrated on one device mis-selects at least two of the four targets, with per-level prediction errors reported between 82% and 365%.

The broader analysis sweeps 40 log-spaced budgets, four reference devices, and four target devices, producing 640 proxy evaluations. Even same-device proportional fits mis-select 37.5--65% of budgets. Transfers from faster to slower targets reach 85--95% mis-selection or violation rates in the worst pairings. The conclusion is limited to this simple proxy, candidate family, and device set. Learned predictors and hardware-aware NAS are not evaluated by this ablation.

### 5.3. Progressive routing ablation before complete overhead

Using candidate-only lookup costs on seed0, D has no violations in 80 device--split--budget cells. A violates 28, B violates 5, C violates 44, and raw entropy violates 20. Among the 52 seed0 cells in which A is feasible, D records 36 wins, 12 ties, and 4 losses, with mean D-minus-A mIoU of +0.0224. The four losses all correspond to ACDC/rain under one mid-range operating pattern. Candidate-specific calibration inversions average 2.13%, suggesting that larger candidates are rarely predicted worse than adjacent smaller ones. The raw probe's area under the risk--coverage curve averages 0.1467 and is worst on ACDC/night at 0.2616, showing that uncertainty ranking weakens under the hardest condition.

Replicating the lookup-table comparison on the other two training runs preserved the qualitative advantage, but these pre-overhead values are not the final headline. The next section replaces candidate-only lookup costs with measured complete route costs and uses corrected, per-run canonical risk grids.

### 5.4. Real route overhead is backend-dependent

**Table 3. Median warm end-to-end route cost used in final replay. Each row includes the tiny probe and policy path.**

| Route | E3 TensorRT/CUDA | E1 Hailo-8 |
|---|---:|---:|
| tiny to tiny | 2.07 ms | 34.95 ms |
| tiny to small | 5.23 ms | 46.13 ms |
| tiny to medium | 11.97 ms | 63.36 ms |
| tiny to large | 26.70 ms | 92.68 ms |

On E3, a naive host-side risk path takes approximately 40--80 ms per call and would erase the candidate-latency advantage. A CUDA entropy kernel reduces complete route latency by 2.5--21 times relative to that path. On 40 real TensorRT outputs, 10 per level, its maximum scalar difference from the production PyTorch risk function is 7.15e-7 nats, below the locked 1e-4 tolerance. The tested ten tiny-probe samples produce zero A-policy and zero D-policy decision mismatches. These exact counts establish observed implementation agreement, not a population agreement probability.

On E1, mandatory network-group activation and deactivation dominate. Warm routes are 2--17 times slower than candidate-only streaming values. Cold measurement requires recreating the complete VDevice each iteration because configured groups cannot be individually released and only one VDevice is available. Since this differs from E3's candidate-only reload, cold numbers are reported as backend-specific stress measurements and are not pooled.

### 5.5. Canonical overhead-aware router result

The final replay recomputes A and D decisions with measured warm route costs, constructs the budget grid from the four route costs of each backend, and chooses risk targets from fit-half performance. The same per-run risk grid is used for E1 and E3. Each dataset split retains the calibrators fitted on its own fit half, so these results evaluate condition-specific calibration rather than a single policy that infers an unknown condition.

**Table 4. Final A-versus-D result with measured complete route costs. W/T/L is computed only on fair cells; violations use all cells.**

| Backend | Fair cells | D W/T/L | Win or tie | Macro D-A mIoU | D violating | A violating |
|---|---:|---:|---:|---:|---:|---:|
| E3 TensorRT/CUDA | 35/60 | 26/8/1 | 97.1% | +0.0241 | 0/60 | 25/60 |
| E1 Hailo-8 | 34/60 | 25/8/1 | 97.1% | +0.0241 | 0/60 | 26/60 |
| Overall | 69/120 | 51/16/2 | 97.1% | +0.0241 | 0/120 | 51/120 |

D outperforms or matches A in 67 of 69 fair cells. The macro gain of 0.0241 mIoU equals 2.41 points on the 0--100 scale. This is a descriptive macro average across the three training runs and two cost tables, not an estimate from 69 independent trials. E1 and E3 reuse the same checkpoint predictions; the cross-backend evidence shows that two very different measured cost mechanisms preserve the policy's qualitative advantage, not independent accuracy replication.

The two losses occur in correlated ACDC/rain cells and do not establish a backend-specific failure rate. Identical decisions across backends are also possible when both policies depend on the ordering of four candidates and the budget grid uses those same four breakpoints. Seed2 is an example: backend results coincide despite very different absolute latencies. Seed3 differs by one fair cell because fit-half mean latency crosses a budget boundary differently. Thus hardware costs condition feasibility and deployed latency, but they do not guarantee that every device chooses a different level.

D's 0/120 violation count follows partly from restricting choices to candidates within budget. The substantive result is the quality retained under that constraint and the contrast with A, whose fit-selected rank heuristic creates 51 violating cells. Because fair cells are defined by A's feasibility, the 67/69 quality statement must always be reported with the all-cell violation counts.

### 5.6. EMA-percentile fake-quant QAT

Dynamic fake quantization of the shared large level has worst-condition losses of 1.66, 2.54, and 1.39 mIoU points for the three tested runs, and two runs exceed the predeclared 1.5-point upper bar. Maximum-observer calibration makes seed0 worse on every split. The 2x2 seed0 screen shows that changing to EMA-percentile calibration improves worst-condition degradation more than exporting the subnet, while mean effects are not uniformly positive. Because the maximum over conditions can change identity between cells, these differences are not interpreted as additive factorial causal effects.

**Table 5. Absolute worst-condition mIoU degradation under shared-supernet EMA-percentile fake-quant QAT.**

| Training run | tiny | small | medium | large |
|---|---:|---:|---:|---:|
| seed-labelled run 0 | 0.32 | 0.97 | 0.51 | 0.97 |
| seed-labelled run 3 | 0.36 | 0.33 | 0.56 | 0.86 |
| seed-labelled run 2 | 0.69 | 0.33 | 0.49 | 1.38 |

Every evaluated run and level remains within 1.5 points. On the previously worst large-level run, the bound improves from 2.54 to 1.38 points. The supported statement is that this observer stabilizes worst-condition fake-quant degradation in the tested setting. Mean quality is not consistently improved, activation outlier suppression is a plausible but unmeasured mechanism, and compiled INT8 accuracy is unknown.

## 6. Discussion

The results separate three decisions that are often conflated. Elastic training determines which candidates exist. Hardware measurement determines their cost and the feasible set under a deployment budget. Calibrated routing decides which feasible candidate should process an image. PACE-Seg's strongest evidence concerns the third decision after the second is measured directly.

The FLOPs experiment explains why the separation matters. FLOPs correctly orders the four levels here but badly exaggerates spacing and cannot reproduce all device-specific budget choices. D does not require a globally accurate latency model; it needs a small measured table of complete route costs. This simplicity is useful for four candidates, although it would become costly for a much larger architecture space.

Within each evaluated domain or condition, candidate-specific calibration improves on a single probe-to-rank heuristic because the same entropy can imply different expected errors for different candidates. The low inversion rate indicates that the mappings generally preserve the expected quality ordering. Yet the high AURC on ACDC/night shows that calibration cannot create information absent from the probe. Moreover, selecting the appropriate per-condition calibrator is assumed rather than learned. A pooled calibrator, an explicit condition detector, a richer score, or temporal evidence may relax that assumption, but none is evaluated here.

The overhead study changes the interpretation of hardware-aware routing. On E3, moving uncertainty computation to the accelerator is necessary; on E1, that optimization is unavailable and engine activation dominates. A candidate-only lookup table therefore omits costs that are both large and architecture-dependent. Measuring a route as one trace avoids double-counting and exposes these mechanisms.

The QAT study provides a cautionary parallel. The initial dynamic method and maximum observer support different conclusions from the final EMA-percentile observer. Reporting only an average would have hidden large adverse-condition degradation, while calling the result INT8 deployment would have confused simulated quantization with a compiled engine. The final claim is consequently secondary and explicitly limited to fake quantization.

## 7. Limitations and threats to validity

First, calibration is domain- and condition-specific. Cityscapes and each ACDC condition use calibrators fitted on their own fit halves. The evaluation therefore assumes that the deployment domain or adverse condition is known or configured; it does not demonstrate a pooled calibrator or automatic selection under an unknown condition. The router calibration feature also differs between fitting and deployment: fit-half entropy excludes ground-truth ignore pixels, whereas held-out and runtime entropy include all pixels. Held-out labels do not enter policy selection, so this is not test-label leakage, but it is a supervised feature mismatch that may shift calibrator bins. A fully deployment-matched rerun should compute both fit and inference entropy without the ignore mask.

Second, the evaluation uses alternating halves of validation splits rather than an external test set. Multiple budgets and thresholds reuse the same images and predictions. Operating cells are therefore correlated, and the 120-cell denominator is not a statistical sample size. The three checkpoint runs are the main training-replication axis, but complete immutable training provenance, checkpoint hashes, initialization parents, and proof of identical configurations are incomplete.

Third, fair-cell conditioning is descriptive. It excludes A's budget-violating comparisons so quality is compared only when A honors its contract, but the denominator depends on baseline behavior. The full-grid violation counts are required beside the 67/69 result. D's zero violations are partly guaranteed by construction and should not be interpreted as empirical reliability.

Fourth, E1 and E3 share segmentation predictions. The hardware evidence consists of separately measured complete route-cost tables combined with the same per-image prediction caches. Compiled engines used for timing have the correct architecture but are not the trained engines used to produce accuracy. The study therefore validates cost accounting and overhead-aware replay, not bit-exact compiled-model accuracy or a continuously running integrated application.

Fifth, breakpoint budgets can reduce hardware conditioning to latency rank. Absolute costs change feasibility and mean-latency operating-point selection, but identical orderings can produce identical route choices. The term hardware-cost-conditioned describes the policy input; it does not imply that decisions must differ across devices.

Sixth, QAT uses shared per-layer calibration buffers across levels and fake quantization in PyTorch. No compiled INT8 accuracy, integer-kernel speedup, or backend-specific quantization behavior is established. The observed advantage of EMA-percentile calibration is consistent with reduced outlier sensitivity but does not measure activation distributions directly.

Seventh, no external power meter was available, so the paper makes no energy claim. DLA executes only the encoder because of a 16-subgraph hardware limit and is not a headline backend. Temporal-window routing, UIoU, cold-frontier replay, and a decomposition of E3's naive host risk path remain future work. ACDC/night's weak probe signal is an observed limitation, not evidence of universal adverse-condition robustness.

Finally, the supernet's compiler-safe operator set was smoke-tested but not compared with a trained unconstrained search space. Compiler portability should therefore be read as an engineering constraint and validation outcome, not an independently demonstrated algorithmic contribution.

## 8. Conclusion

PACE-Seg combines an elastic segmentation model with per-domain candidate-specific calibration and measured-budget runtime routing. In the evaluated four-candidate system, measured accelerator latency differs substantially from a simple proportional FLOPs proxy, while the largest elastic level remains near a matched independently trained model across clean and adverse conditions. After complete warm route costs are measured on TensorRT/CUDA and Hailo-8, offline per-image replay shows that the constrained candidate-specific policy outperforms or matches the rank baseline in 67 of 69 fair cells, gains 0.0241 mIoU on average, and satisfies every evaluated budget where the rank baseline creates 51 violating cells. These counts are correlated, the backends share predictions, zero violations partly follow from the policy design, and each condition uses its own fit-half calibrators. The study therefore supports a bounded conclusion: directly measured route costs and candidate-specific error calibration can improve runtime selection among elastic segmentation candidates when the deployment domain is known, but they do not establish condition-agnostic routing, universal reliability, compiled INT8 accuracy, energy savings, or independent cross-device accuracy replication.

## Data and code availability

The repository records the implementation, experiment reports, and canonical result artifacts. The immutable authoring snapshot used for this manuscript is commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`; the evidence package identified inside that snapshot is based on commit `1c4d42a7ea525931154ebb2f9015d898ef7771a5`. The final overhead-aware router evidence is stored in `reports/router_overhead_replay_E3_20260922.json`, `reports/router_overhead_replay_E1_20260928.json`, `reports/router_overhead_replay_E3_seed2_20260929.json`, `reports/router_overhead_replay_E1_seed2_20260929.json`, `reports/router_overhead_replay_E3_seed3_20260929.json`, and `reports/router_overhead_replay_E1_seed3_20260929.json`, together with `reports/router_overhead_v1_20260922.md` and `reports/audit_gpu_risk_kernel_production_E3.json`. Dataset access remains subject to the Cityscapes and ACDC licenses.

## References

1. Yu, C., Wang, J., Peng, C., Gao, C., Yu, G., Sang, N. BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation. *ECCV*, 2018. https://openaccess.thecvf.com/content_ECCV_2018/html/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.html
2. Fan, M., Lai, S., Huang, J., Wei, X., Chai, Z., Luo, J., Wei, X. Rethinking BiSeNet for Real-Time Semantic Segmentation. *CVPR*, 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.html
3. Xu, J., Xiong, Z., Bhattacharyya, S. P. PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers. *CVPR*, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html
4. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*, 2021. https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
5. Yu, J., Yang, L., Xu, N., Yang, J., Huang, T. Slimmable Neural Networks. *ICLR Workshop*, 2019. https://arxiv.org/abs/1812.08928
6. Yu, J., Huang, T. Universally Slimmable Networks and Improved Training Techniques. *ICCV*, 2019. https://ieeexplore.ieee.org/document/9009445
7. Cai, H., Gan, C., Wang, T., Zhang, Z., Han, S. Once-for-All: Train One Network and Specialize it for Efficient Deployment. *ICLR*, 2020. https://openreview.net/pdf?id=HylxE1HKwS
8. Xue, D., Yang, F., Wang, P., Herranz, L., Sun, J., Zhu, Y., Zhang, Y. SlimSeg: Slimmable Semantic Segmentation with Boundary Supervision. *ACM Multimedia*, 2022. https://doi.org/10.1145/3503161.3548191
9. Kouris, A., Venieris, S. I., Laskaridis, S., Lane, N. D. Multi-Exit Semantic Segmentation Networks. *ECCV*, 2022. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810326.pdf
10. Li, Y., Song, L., Chen, Y., Li, Z., Zhang, X., Wang, X., Sun, J. Learning Dynamic Routing for Semantic Segmentation. *CVPR*, 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Learning_Dynamic_Routing_for_Semantic_Segmentation_CVPR_2020_paper.html
11. Liu, Z., Xu, Z., Wang, H.-J., Darrell, T., Shelhamer, E. Anytime Dense Prediction with Confidence Adaptivity. *ICLR*, 2022. https://openreview.net/pdf?id=kNKFOXleuC
12. Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., Le, Q. V. MnasNet: Platform-Aware Neural Architecture Search for Mobile. *CVPR*, 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html
13. Chen, W., Gong, X., Liu, X., Zhang, Q., Li, Y., Wang, Z. FasterSeg: Searching for Faster Real-time Semantic Segmentation. *ICLR*, 2020. https://arxiv.org/abs/1912.10917
14. Dou, Z., Ye, D.-Y., Wang, B., Wang, X., Sun, S. AutoSegEdge: Searching for the Edge Device Real-Time Semantic Segmentation Based on Multi-Task Learning. *Image and Vision Computing* 136 (2023) 104719. https://doi.org/10.1016/j.imavis.2023.104719
15. Dou, Z., Ye, D.-Y., Wang, B., Wang, X., Sun, S. Multi-Objective Neural Architecture Search for Efficient and Fast Semantic Segmentation on Edge. *IEEE Transactions on Intelligent Vehicles* 9(1), 2024, 1346--1357. https://doi.org/10.1109/TIV.2023.3332594
16. Li, Y., Yang, C., Zhao, P., Yuan, G., Niu, W., Guan, J., Tang, H., Qin, M., Jin, Q., Ren, B., Lin, X., Wang, Y. Towards Real-Time Segmentation on the Edge. *AAAI*, 2023. https://doi.org/10.1609/aaai.v37i2.25232
17. Wang, D., Gong, B., Wang, L. On Calibrating Semantic Segmentation Models: Analyses and an Algorithm. *CVPR*, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Wang_On_Calibrating_Semantic_Segmentation_Models_Analyses_and_an_Algorithm_CVPR_2023_paper.html
18. de Jorge, P., Volpi, R., Torr, P. H. S., Rogez, G. Reliability in Semantic Segmentation: Are We on the Right Track? *CVPR*, 2023. https://openaccess.thecvf.com/content/CVPR2023/html/de_Jorge_Reliability_in_Semantic_Segmentation_Are_We_on_the_Right_Track_CVPR_2023_paper.html
19. Sakaridis, C., Wang, H., Li, K., Zurbrugg, R., Jadon, A., Abbeloos, W., Reino, D. O., Van Gool, L., Dai, D. ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding. *ICCV*, 2021. https://ieeexplore.ieee.org/document/9711067
20. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B. The Cityscapes Dataset for Semantic Urban Scene Understanding. *CVPR*, 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html
21. Xu, S., Li, Y., Lin, M., Gao, P., Guo, G., Lue, J., Zhang, B. EQ-Net: Elastic Quantization Neural Networks. *ICCV*, 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Xu_EQ-Net_Elastic_Quantization_Neural_Networks_ICCV_2023_paper.html
22. Jin, Q., Yang, L., Liao, Z. AdaBits: Neural Network Quantization With Adaptive Bit-Widths. *CVPR*, 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.html
