# Experiment provenance — ARR resubmission cycle (Sept 2026)

> **Framing (2026-09-04):** the paper presents the hard-concrete gate as *the* method. There is
> no "corrected method", no before/after arm, no diagnosis section — new reviewers never see the
> old reviews. Everything below is internal history for us, not paper content.

Nothing here is deleted. `data/`, `docs/`, `figures/`, `analysis/` and `lightning_logs/`
are all gitignored, so these artifacts have **no git backup** — treat deletion as
irreversible. This file records what each superseded artifact was for, so it stays
interpretable if we need to reference it later.

## Superseded: the visibility-filter experiment

**Idea.** MediaPipe's BlazePose regresses all 33 pose landmarks whether or not they are
inside the frame, so everything below the hips is fabricated in these corpora
(never-filmed: 6 landmarks on AUTSL, 8 on GSL, 10 on ASL Citizen, at >80% out-of-frame).
The old annealed-sigmoid gate ranked 7 of GSL's 8 never-filmed landmarks inside the
top-48. We built a filter to drop them from the candidate pool before top-K selection.

**Why it was retired (2026-09-03).** A correctly configured hard-concrete gate demotes
them on its own — on GSL, median phantom rank moves 35 → 132 and top-48 count 7 → 0 at
`init_keep_probability: 0.5`. Fixing the gate addresses the cause; the filter edited the
output and required a hand-tuned 0.8 threshold to defend. The filter also never touched
stage-1 or normalization, so fabricated coordinates still shaped the mean/std that scale
every kept landmark (`src/data/asl_dataset.py` normalizes over all 543 at lines 192-206,
then subsets at line 232). That normalization leak is still open.

**Artifacts, all retained:**

| path | what |
|---|---|
| `data/{,gsl/,asl_citizen/}visibility_mask.json` | old masks; superseded by `landmark_visibility.json` (renamed keys, diagnostic only) |
| `data/*/informed_selection_visfilt/topk/` | filtered top-K subsets |
| `configs/visfilt/` | 54 stage-2 configs (3 datasets x 6 K x 3 seeds); only the 2 GSL ones below were ever run |
| `scripts/generate_visfilt_configs.py` | generator. **Worth repurposing** for the seeded stage-2 validation runs — point it at the new gate's rankings |
| `archive/SLR/models/gsl/informed_selection_visfilt/` | 2 completed GSL runs + test metrics |

**Result (GSL, seed 4000, filtered vs original subset):**

| K | original test | filtered test | val delta |
|---|---|---|---|
| 100 | 67.31 | 67.97 (+0.66) | -3.4 |
| 48 | 70.54 | 66.54 (-4.00) | -5.3 |

**Inconclusive at n=1** — K=100 flipped sign between val and test. This is the direct
demonstration that single-seed comparisons cannot resolve effects of this size, which is
also what all four reviewers said. Everything comparative needs >=3 seeds from here.

Jobs: `13504939` (train, K=100 on dw), `13506039` (train, K=48 on cs), `13549373` (test).
Two earlier test attempts (`13518895`, `13544296`) FAILED — see the DDP note below.

## Gate sweep 1 — `13549393`, 4 runs, `archive/SLR/models/gsl/gate_sweep/`

First hard-concrete runs. Judged on the wrong criterion at first (gate closure); the
criterion that matters is the quality of the ranking distribution, since stage-2 does the
discrete selection anyway.

- `num_exact_zero` = 0 in all four, but `num_exact_one` reached 44-52, so the
  stretch-and-clamp works in real training. The old sigmoid gate can produce neither.
- Lambda only reached 18-34% of target: `l0_warmup_steps: 10000` + `l0_anneal_steps: 40000`
  puts full strength at step 50k, but every run early-stopped by step 21.7k.
- **Early stopping on val_acc is the wrong criterion for stage-1**, whose product is the
  ranking, not accuracy. `hc_lam2_init0.5` had the worst val_acc (0.5642) and the best
  ranking.
- **Init dominates lambda.** At init 0.9 gates sit at open-prob 0.978 and stay pinned
  (IQR 0.0012-0.0017, worse than the old gate's 0.0135). At init 0.5: IQR 0.0116,
  displacement confound halved (rho 0.690 -> 0.347) and nearly gone within the face mesh
  (0.565 -> 0.097).
- Face landmarks in top-48: **still 0 in every config.** Not a gate problem — it is the
  global normalization scale plus 468-way redundancy in the face mesh.

## Reverted code

`src/selection/select_top_k_joints.py` had an `--exclude-indices` flag; removed 2026-09-03,
file is back to its committed state. `src/pose/compute_visibility_mask.py` was renamed to
`src/pose/measure_landmark_visibility.py` and reframed as diagnostic-only.

## Gotcha found while debugging

TEST mode needs a **1-task** allocation. Lightning takes `world_size` from `SLURM_NTASKS`,
so an 8-task allocation opens a DDP rendezvous and blocks 30 min waiting for peers a single
`python` process never spawns; setting `n_gpus: 1` alone does not prevent it. Use
`sbatch/test_visfilt_array.sh`. `sbatch/test_informed_selection.sh` requests
`--ntasks-per-node=4` with `--gpus=1`, which looked like the same latent bug.

**SPOT-CHECKED 2026-09-06 — CLEARED.** All 27 submitted-paper test runs are sound:
- Every predictions CSV covers the FULL test split (AUTSL 3,742 / ASL Citizen 32,941 /
  GSL 3,500 rows, exact). No DDP sharding truncation, which was the failure mode that would
  have silently computed accuracy on 1/N of the data.
- Accuracy recomputed from the raw per-clip predictions matches the recorded `metrics.json`
  exactly for all 27 runs (largest discrepancy 0.00e+00).
- The two independently-computed fields `accuracy` and `my_accuracy` agree to 7 decimals.

The 4-task allocation is still wrong in principle and new TEST passes should use the 1-task
`test_visfilt_array.sh`, but **no published number is contaminated**. This also confirms
reviewer pa2X's C1: the metric is micro accuracy (correct / total).

## Gate sweep 2 + baselines (2026-09-04) — the runs that count

**Gate config settled:** `gate_type: hard_concrete`, `init_keep_probability: 0.5`,
`l0_end_weight: 2.0`, `l0_warmup_steps: 0`, `l0_anneal_steps: 5000`, `early_stop: 1000`,
`max_steps: 40000`, `n_gpus: 2`. Jobs `13581750` / `13581932`, results in
`archive/SLR/models/gsl/gate_sweep_2gpu/`.

**TSLFormer hand-crafted baseline: 9/9 complete** (jobs `13581751`, `13581933`), configs in
`configs/tslformer/`, outputs in `archive/SLR/models/{,gsl/,asl_citizen/}tslformer_2gpu/`.
Validation only — still needs TEST passes.

**Stage-1 rankings** for AUTSL + ASL Citizen: job `13582020`, `configs/gate_stage1/`,
outputs in `archive/SLR/models/{,asl_citizen/}gate_stage1/hc_init0.5_lam2/`.

**Infrastructure fixes applied this session** (see the throughput memory for numbers):
DataLoader workers (`num_workers` default 6), vectorized `interpolate_with_gaps` (bit-identical),
`--mem` sized per-rank (20 GiB/rank, 48 GiB floor) instead of a flat 1000 GiB, `--cpus-per-task=8`,
val-skeleton cache (correct but ~neutral for speed), and `%x`-first log naming so
`clean_slurm_outputs.py` stops deleting live jobs' logs. Net: ~4,335 -> ~54,000 steps/h.

**Paper draft** lives in `docs/emnlp_v2.tex`; `docs/emnlp.tex` is the submitted record, with a
timestamped copy under `docs/.backups/`.

## Normalization scale distortion — measured 2026-09-04

`src/data/asl_dataset.py` computes mean/std over axis `(0,1)` of the `(T,543,2)` array
(lines ~205-231), i.e. over all frames **and all 543 landmarks**, and only subsets to the
selected K at line 258. Since the face mesh is 468/543 = 86% of the landmarks, the global
statistics are dominated by the face; but the *scale* is set by whatever sprawls most.

Measured on 150 random val clips per dataset (`sd` over valid, non-sentinel, finite coords):

| dataset | global std (x, y) | face y-scale | pose y-scale | face:pose ratio |
|---|---|---|---|---|
| AUTSL | 0.0437, 0.1520 | 5.23x compressed | 0.44 (2.3x amplified) | 11.9x |
| ASL Citizen | 0.0735, 0.3293 | 4.35x compressed | 0.36 (2.8x amplified) | 12.1x |
| GSL | 0.0442, 0.1917 | 5.13x compressed | 0.39 (2.6x amplified) | 13.2x |

"y-scale" = global std / that group's own std, i.e. the factor by which global normalization
shrinks the group's motion relative to normalizing on its own statistics. >1 compressed.

**CORRECTION (same day).** An earlier version of this note claimed global normalization hands
pose landmarks a ~12x advantage over face landmarks in the quantity the gate ranks by. That is
wrong. The code computes ONE mean and ONE std per axis and applies them to all 543 landmarks, so
the mean cancels in any frame-to-frame difference and the std is a common factor: **global
normalization cannot change the relative displacement ordering of landmarks.** The table above is
a *prediction of what per-group normalization would do*, not a description of the current bias.

There are two separate problems, which that note conflated:

**Problem 1 - the selected subset arrives mis-scaled.** Normalizing over 543 and then subsetting
means the kept landmarks are centered/scaled by statistics from discarded landmarks. Measured on
the real `top_24_indices.json` subset over 120 AUTSL val clips, what reaches the encoder is
`mean = (+1.169, +1.569)`, `std = (2.206, 1.164)` instead of `(0,0)` / `(1,1)` -- off-center by
over 1 sd and over-dispersed 2.2x in x. This is an accuracy/optimization issue and is exactly
reviewer ABDG's "distorts their input distribution". It does **not** bias the stage-1 ranking.
Tested by the 9-run stage-2 subset-statistics ablation.

**Problem 2 - the face genuinely barely moves.** Mean per-frame |displacement| in raw image units
over 120 AUTSL val clips: upper lip 0.00093, mouth corner 0.00093, right wrist 0.00692, left wrist
0.01468, right index tip 0.00874. Mean face landmark 0.00097 vs mean pose landmark 0.01304 --
**13.5x, present in the raw data before normalization**. A displacement-ranking gate deprioritizes
face landmarks because they really do move 13.5x less. The flawed inference is "moves little" =>
"uninformative"; mouthing can be the sole distinguishing feature between two signs (BSL1K,
Albanie et al.). Tested by the 3-run stage-1 per-group/per-landmark standardization experiment,
which gives each landmark its own scale so it is judged against its own variation. **This is the
experiment that decides whether "the face is uninformative" is a finding or an artifact of the
criterion.**

**Context.** The gate's ranking correlates with displacement magnitude (Spearman rho = 0.639 at
`init 0.5 / lam 2`), which is why the raw amplitude gap in Problem 2 propagates straight into the
ranking. Note also that the pose block's large own-std is partly inflated by the fabricated
out-of-frame leg landmarks sprawling across the frame.

**Runs to launch:** (1) stage-2 subset statistics, K in {48,24,10} x 3 datasets x 1 seed = 9 runs;
(2) stage-1 per-group standardization, 1 per language = 3 runs.

## AUTSL stage-1, new gate vs old — job 13582020_0, completed 40,000 steps

`models/gate_stage1/hc_init0.5_lam2` vs old `models/pruning_sweep_l0_fixed_v3_optimized_v2`.
Open probs from `last.ckpt` via `sigmoid(logits - beta*log(-gamma/zeta))`; gate values via
stretch-and-clamp of `sigmoid(logits)`.

**The gate now closes.** 309 of 543 gates are *exactly* zero (234 open). The old gate's
minimum probability across all 543 landmarks was **0.693** — it never pushed any landmark
below two-thirds open. Hard-closed by group: face 65.2%, pose 12.1%, LH 0%, RH 0%.
This also means the method now yields an intrinsic K (~234 here) rather than only a ranking.

**The AUTSL ranking barely moved.** Spearman(old, new) = +0.805; top-48 overlap 46/48 with
identical composition (face 0 / pose 10 / LH 21 / RH 17); top-24 overlap 21/24; top-10 9/10.
IQR 0.0984 -> 0.1058 (only 1.1x, vs GSL's 9x). Never-filmed landmarks [495-500] were already
outside the old top-48 (median rank 58 -> 56). **The published AUTSL numbers hold up** — GSL
was the dataset with the artifact problem, as expected from its 7-of-8 phantoms in top-48.

**Displacement confound — the aggregate statistic is misleading.** Over all 543 landmarks
rho drops 0.282 -> 0.096, which looks like the confound vanished. Decomposed it has not:

| level | rho(open-prob, mean abs displacement) |
|---|---|
| across the 4 group means (n=4, illustrative) | +0.80 |
| within hands (501-543) | +0.705 |
| within face mesh (0-468) | **-0.350** |

The face mesh is 86% of the landmarks and its internal correlation is *negative*, which cancels
the positive between-group correlation in the 543-wide aggregate. **Report the decomposition, not
the aggregate.** Amplitude drives the ranking *between* anatomical groups (which is what excludes
the face wholesale) and *within the hands*, but not within the face.

Group medians (rank / mean open-prob / mean per-frame |displacement|):
LH 12 / 0.897 / 0.0156 · RH 36 / 0.843 / 0.0069 · pose 58 / 0.601 / 0.0084 · face 306 / 0.289 / 0.00088

## Normalization-scope experiment — job 13582035, launched 2026-09-04

**Code.** `src/data/asl_dataset.py` gained a `normalization_scope` option ("global" |
"per_group" | "per_landmark"), refactored out of the inline block into `_standardize()`.
`src/train.py` passes `config.get("normalization_scope", "global")` at all four dataset
construction sites (train / val / test / TTA).

**The `global` path is bit-identical to the previous inline code** (verified over 40 AUTSL
val clips, max abs diff 0.000e+00), so every prior run stays reproducible.

Effect on post-standardization group std, one AUTSL clip:

| scope | face | pose | LH | RH |
|---|---|---|---|---|
| global | 0.542 | 2.183 | 1.286 | 2.502 |
| per_group | 1.000 | 1.000 | 1.000 | 1.000 |
| per_landmark | 1.000 | 1.000 | 1.000 | 1.000 |

**Runs.** `configs/gate_norm/{autsl,asl_citizen,gsl}_pergroup.yaml`, identical to the settled
stage-1 config (hard_concrete, init 0.5, lam 2, 40k steps, 2 GPUs, seed 4000) except for the
scope and the save dir -> `models/gate_norm/*_pergroup`. Job `13582035`, array 0-2%3 on cs,cs3.

**The question it answers.** Does the gate select face landmarks once the between-group
amplitude gap is removed? If yes, "no face landmarks in the top-48" is an artifact of ranking
by motion magnitude and per-group standardization becomes the preprocessing we ship. If no, it
is a real limitation of the criterion and goes in Limitations with the amplitude table as
evidence. Either way we can say what we looked at.

## ASL Citizen stage-1, new gate vs old — job 13582020_1, 40,000 steps, 02:01:24

Old 543-wide reference: `models/asl_citizen/s/figures/joint_probabilities.csv`.

| | OLD | NEW |
|---|---|---|
| IQR | 0.0289 | 0.2195 (**7.6x**) |
| min / max | 0.791 / 0.914 | 0.126 / 0.944 |
| gates exactly 0 | 0 | **250** |
| Spearman(old, new) | — | **+0.877** |

Top-48 overlap 44/48 (composition face 0 / pose 9->10 / LH 18->17 / RH 21->21).
Top-24 overlap 18/24. Top-10 overlap 7/10.
Fabricated landmarks [491-500]: median rank 55 -> 56, in top-48 **1 -> 0**.
Hard-closed by group: face 53%, pose 9%, LH 0%, RH 0%.

### All three languages, summarized

| dataset | IQR change | hard zeros | rho(old,new) | top-48 overlap | phantoms in top-48 |
|---|---|---|---|---|---|
| AUTSL | 0.0984 -> 0.1058 (1.1x) | 309 | +0.805 | 46/48 | 0 -> 0 |
| ASL Citizen | 0.0289 -> 0.2195 (7.6x) | 250 | +0.877 | 44/48 | 1 -> 0 |
| GSL | 0.0135 -> 0.1186 (9x) | (see gate_sweep_2gpu) | — | — | 7 -> 0 |

**Two readings that matter for the paper.** (1) The old gate could not close *anything* on any
dataset — minima were 0.693 (AUTSL), 0.791 (ASL Citizen). The new gate hard-closes 250-309 of 543
and yields an intrinsic K. (2) Rankings are stable at K=48 (44-46/48) but diverge at K=24
(18-21/24) and K=10 (7-9/10). **The tail of a single 543-wide ranking is its least reliable part,
which is the empirical case for the iterative cascade** re-ranking within each reduced pool.

## Per-group standardization results — jobs 13582035 / 13582044, all COMPLETED at 40,000 steps

**The face does not come back. Face exclusion is not an amplitude artifact.**

| dataset | scope | top-48 composition | best face rank | face in top-100 | face in top-270 |
|---|---|---|---|---|---|
| AUTSL | global | face 0 / pose 10 / LH 21 / RH 17 | 64 | 35 | 198 |
| AUTSL | per_group | face 0 / pose 10 / LH 21 / RH 17 | 60 | 28 | 195 |
| ASL Citizen | global | face 0 / pose 10 / LH 17 / RH 21 | 64 | 36 | 199 |
| ASL Citizen | per_group | face 0 / pose 8 / LH 19 / RH 21 | 62 | 32 | 195 |
| GSL | global | face 0 / pose 9 / LH 18 / RH 21 | 63 | 35 | 196 |
| GSL | per_group | face 0 / pose 8 / LH 19 / RH 21 | 50 | 35 | 196 |

Chance baseline: a random ranking puts 468/543 = 86.2% of any prefix in the face, so ~86 of
top-100 and ~233 of top-270. Face is *under*-represented against chance at every depth, under
both scopes. Face median rank is 306-308 in all six runs.

**Reading.** Equalizing the amplitude scale between anatomical groups (verified to put all four
groups at exactly unit variance) moved the best face landmark from rank 63-64 to 50-62 and
changed nothing about top-48 composition. The ~75 non-face landmarks (33 pose + 42 hand)
monopolize the first ~50-64 slots regardless. So the earlier hypothesis -- that the face is
excluded because it moves ~13x less and the gate ranks by amplitude -- **is not supported**.
Remaining candidate explanations: 468-way redundancy within the face mesh (dropping any single
face landmark costs almost nothing because 467 neighbours carry the same information), and
genuinely low lexical relevance for these three datasets' vocabularies. This is now a
defensible Limitations statement rather than an untested artifact claim.

**Stage-1 val accuracy (n=1 seed, NOT resolvable at this effect size):**
AUTSL 0.8183 global vs 0.7951 per_group (-2.3pp); ASL Citizen 0.5163 vs 0.5369 (+2.1pp);
GSL 0.5972 vs 0.6635 (+6.6pp). Mixed. Do not claim per_group helps or hurts without seeds.

**Cost:** per_group is free in wall clock -- AUTSL 01:18:09 vs 01:17:51 for global on 2xB200.

## CORRECTION: 1 GPU is NOT free — the jobs are dataloader-bound

Same GSL config, 40,000 steps, A100:

| alloc | CPUs | dataloader workers | elapsed | billing | billing x hours |
|---|---|---|---|---|---|
| 2 GPU (`13581932_0`) | 16 | 2 ranks x 6 = 12 | **01:01:11** | 448 | 457 |
| 1 GPU (`13582044_0`) | 8 | 1 rank x 6 = **6** | **02:14:40** | 224 | 503 |

1 GPU was **2.20x slower** -- almost exactly the 2x ratio in worker count. The earlier claim that
"total work per step is unchanged so 1 GPU costs nothing" was **wrong**: `batch_size =
effective_batch_size / n_gpus` keeps per-step work constant, but `--cpus-per-task=8` is
*per task*, so halving the ranks also halves the CPU workers feeding the GPU. These runs are
CPU/dataloader-bound, not GPU-bound -- consistent with 2,098 MiB of 183,359 MiB GPU memory in use
and one rank idling at 0-15%.

**The efficient configuration is 1 GPU with the CPUs of two:** `--gpus=1 --cpus-per-task=16` and
`num_workers=14`. That halves GPU occupancy (the scarce resource) while keeping the resource that
actually sets throughput. Untested -- measure with a short-`max_steps` probe before committing the
~176-run queue.

### Throughput probe results — the jobs are worker-bound, not GPU-bound

Median training it/s (= optimizer steps/s; both configs process 64 clips per step, so this is
apples-to-apples), same GSL config, same A100 hardware. Cross-checked against wall clock:
40,000 / 10.96 s^-1 = 60.8 min vs an actual 61:11 elapsed.

| alloc | CPUs | workers | it/s | vs 2-GPU | billing | billing x h per 40k steps |
|---|---|---|---|---|---|---|
| 2 GPU (`13581932_0`) | 16 | 12 | 10.96 | 1.00x | 448 | 457 |
| 1 GPU (`13582044_0`) | 8 | 6 | 5.29 | 0.48x | 224 | 503 |
| **1 GPU (`13582270_0`)** | **16** | **14** | **12.02** | **1.10x** | **224** | **207** |

Scaling is near-linear in **worker count** and indifferent to GPU count: 6->12 workers = 2.07x,
6->14 = 2.27x. The GPU was never the constraint (2,098 MiB of 183,359 MiB used, one rank idling
at 0-15%) -- it was starved.

**Default from 2026-09-04: `--gpus=1 --cpus-per-task=16` with `num_workers: 14`.** Faster than the
2-GPU config, half the GPUs, and 2.2x cheaper in billing (which is what AssocGrpBillingRunMinutes
caps). Probe artifacts under `models/probe/`, configs under `configs/probe/`, throwaway.

### 32-CPU probe (`13582292_0`) and the final decision

| alloc | CPU | workers | it/s | billing | billing x h / 40k | jobs/node | node it/s |
|---|---|---|---|---|---|---|---|
| 2 GPU | 16 | 12 | 10.96 | 448 | 457 | 4 | 43.8 |
| 1 GPU | 8 | 6 | 5.29 | 224 | 503 | 8 | 42.3 |
| **1 GPU** | **16** | **14** | **12.02** | **224** | **207** | **8** | **96.2** |
| 1 GPU | 32 | 30 | 22.40 | 448 | 223 | 4 | 89.6 |

SLURM bills CPUs as well as GPUs -- 32 cores + 1 GPU costs the same 448 as 2 GPUs + 16 cores.
So 32 CPU doubles single-job speed but halves jobs-per-node and costs 7% more per unit work.

**Decision: `--gpus=1 --cpus-per-task=16` for the bulk queue** (54 top-K stage-2, 45 random-K,
45 cascade stage-2, ~120 TEST passes) -- these are independent, so aggregate node throughput is
what matters and 16 CPU wins it. **Use 32 CPU only for the cascade chains**, which are sequential
and latency-bound: only 3 run at once (one per language) and halving per-stage wall clock directly
shortens the critical path, currently ~16.5 h on ASL Citizen.

**Code hardening.** `src/train.py` now has `_default_num_workers()`, deriving workers from
`SLURM_CPUS_PER_TASK - 2` (falling back to 6 off-SLURM); an explicit `num_workers:` in the config
still wins. Verified: 8 -> 6, 16 -> 14, 32 -> 30, unset/garbage -> 6. This closes the trap that
produced the wrong conclusion above -- worker count can no longer silently drift from the CPU
allocation when the rank count changes.

## Region ablation — jobs 13582329 (dw) / 13582330 (cs), launched 2026-09-04

**Question.** The gate selects zero face landmarks at any K<=48 on all three languages, and
per-group amplitude equalization did not change that. Does the face carry usable signal that the
L0 ranking fails to surface, or is it genuinely uninformative for these vocabularies?

**Design.** Train the encoder on one anatomical region only, no gate (`joint_pruning: False`;
the dataset subsets via `joint_indices_file`, and `build_skeleton_encoder` sizes from
`num_pose_points` independently of the gate). Same recipe as the upcoming stage-2 queue:
seed 4000, `max_steps: 40000`, `early_stop: 30`, `n_gpus: 1`, `normalization_scope: global`
(the shipped default, not the per_group probe).

| region | n | indices |
|---|---|---|
| face | 468 | `data/region_subsets/face_468_indices.json` (0-467) |
| lips | 40 | `data/region_subsets/lips_indices.json` (FACEMESH_LIPS outer+inner contours) |

Configs `configs/region_ablation/*.yaml`, outputs `models/{,asl_citizen/,gsl/}region_ablation/`.
Lips-only is included because the claim we need to make in Limitations cites the lip-reading
literature specifically; 40 landmarks is a sharper test of that than the whole 468-point mesh.

**Interpretation set in advance, so the result is not read post hoc:**
- Face-only well above chance => the region carries signal the ranking fails to surface. Cause is
  468-way redundancy: dropping any single face landmark barely moves the loss because 467
  neighbours encode the same thing, so an input-level L0 penalty prices each one near zero even
  though the group is collectively informative. Fix would be group-structured sparsity.
- Face-only near chance on all three => these balanced isolated citation-form vocabularies are
  largely separable without the face. Must NOT be generalized to continuous signing, where
  mouthing and grammatical non-manuals carry lexical and syntactic contrasts.

Chance baselines: AUTSL 1/226 = 0.44%, ASL Citizen 1/2731 = 0.037%, GSL 1/310 = 0.32%.
Full-skeleton references (old recipe, single seed): AUTSL ~87%, ASL Citizen ~45%, GSL ~70%.

## Partition policy — no cs3 from 2026-09-04 (user directive)

The B200s on cs-3-1 are the wrong hardware for this workload and other students need them for
jobs that can fill them. Evidence: 843-2,098 MiB of 183,359 MiB GPU memory in use (~1%) and
0-25% utilization, measured even on the 468-landmark face-only runs.

**Why the GPU is never the bottleneck here, structurally.** `build_skeleton_encoder` builds a BERT
that flattens landmarks into one per-frame vector -- `x.view(B, T, J*P)` into
`Linear(num_joints*num_coords, 512)` -- so the transformer sees only T~16 tokens over 2 layers
(~7M params) and `J` affects a single small input projection. All the time goes to CPU
preprocessing, ~93% of it NaN interpolation (31.4 ms/clip AUTSL, 19.5 ms GSL). Encoders that WOULD
be GPU-bound -- SPOTER (tokenizes per landmark, sequence = J = 543, attention quadratic in J),
ST-GCN (graph convs over J nodes), the RGB/VideoMAE branch -- were all cut from the paper.

**Subsetting early does not help** (measured, contrary to expectation): interpolation cost tracks
the columns that actually contain NaNs, which are the frequently-occluded hand landmarks we always
keep. 543 -> 31.35 ms, 75 -> 30.43, 48 -> 28.23, 10 -> 9.46. The 468-point face mesh is detected
reliably and is nearly free to preprocess.

**Consequence for the paper:** pruning to 24-48 landmarks does *not* reduce our training wall
clock, because preprocessing dominates and its cost is K-independent. The efficiency gain is in
model size and inference, not training throughput. The meta-review framed this as "a practically
relevant efficiency question", so the compute appendix should say which cost actually falls
rather than let readers assume training gets cheaper.

**Applied:** every sbatch script audited; no `#SBATCH --partition` targets cs3 any more.
`train_visfilt_2gpu.sh` defaulted to cs3 and now defaults to dw+matrix (and is marked DEPRECATED
in favour of the 1-GPU/16-CPU script). Usage examples now read dw first, then cs. The two GSL
region-ablation jobs already running on cs-3-1 (`13582330`) were left to finish.

## Region ablation RESULTS — 13582329 / 13582330, all COMPLETED

Best val_acc, seed 4000, no gate, `normalization_scope: global`, `max_steps: 40000`.

| dataset | classes | chance | face-only (468) | lips-only (40) | x chance (face) | full skel* |
|---|---|---|---|---|---|---|
| AUTSL | 226 | 0.44% | 14.60% | **16.21%** | 33.0x | ~87% |
| ASL Citizen | 2,731 | 0.037% | 2.49% | **2.91%** | 68.0x | ~45% |
| GSL | 310 | 0.32% | 8.76% | **9.61%** | 27.2x | ~70% |

\* old single-seed recipe, indicative only; being re-run.

**Finding 1 — the face carries real signal.** 27-68x chance on all three languages. The gate
selects **zero** face landmarks at any K<=48. So the exclusion is a failure of the selection
criterion, not an absence of information. This is the pre-registered "face-only well above chance"
branch, and it is the reading reviewer pa2X proposed.

**Finding 2 (unanticipated) — 40 lip landmarks BEAT the whole 468-point face mesh**, on all three
languages, in the same direction every time: +1.61pp AUTSL, +0.42pp ASL Citizen, +0.85pp GSL.
Direct support for the redundancy explanation: the extra 428 face landmarks dilute rather than
add. It also localizes the signal exactly where the sign-language literature says it is (mouthing;
cf. BSL1K, Albanie et al.), which makes the Limitations claim concrete rather than hand-waving.

**Caveats.** Single seed each -- the face-vs-chance gap is enormous and unambiguous, but the
face-vs-lips gaps (0.42-1.61pp) are small and rest on consistency of direction across three
datasets, not on any one comparison. These are val, not test. Full-skeleton references are the old
recipe.

**Possible follow-up (not launched):** matched-budget test -- top-48 UNION lips (88 landmarks)
vs the gate's own top-88. If the union wins at equal K, the gate is demonstrably leaving usable
information on the table. 6 runs.

## MAIN QUEUE LAUNCHED 2026-09-04

### New rankings (hard-concrete gate, global normalization)
Open probs exported from each 543-wide stage-1 `last.ckpt` to
`data/{,asl_citizen/,gsl/}informed_selection_hc/joint_probabilities.csv`, then
`select_top_k_joints.py` for K in {270,100,48,24,10}. Top-48 composition:

| dataset | face | pose | LH | RH |
|---|---|---|---|---|
| AUTSL | 0 | 10 | 21 | 17 |
| ASL Citizen | 0 | 10 | 17 | 21 |
| GSL | 0 | 9 | 18 | 21 |

Face first enters at K=100 (35-36 of 100) and K=270 (196-199 of 270).

### Code: `normalization_scope: "subset"` added
Pools standardization statistics over only the landmarks `selected_joint_indices` retains,
applied at the existing pipeline position so augmentation semantics are unchanged (equivalent
to subsetting first, then standardizing). Verified on the real K=24 subset over 60 AUTSL clips:
`global` delivers mean (+1.195, +1.571) / std (2.129, 1.269); `subset` delivers exactly
(0, 0) / (1, 1). This is reviewer ABDG's normalization objection, addressed directly.

### Stage-2 matrix — `13585389` (dw, 69 runs, %10) and `13585390` (cs, 45 runs, %6)
Generated by `scripts/generate_stage2_configs.py`; 114 configs, 114 distinct save dirs,
collision-checked and all validated (index counts vs `num_pose_points`, index ranges, no
duplicates, all CSVs resolve). Disjoint lists per partition -- never mirror training runs.

| family | runs | design |
|---|---|---|
| topk | 54 | K in {543,270,100,48,24,10} x seeds {4000,4001,4002} x 3 langs |
| randomk | 45 | K in {48,24,10} x 5 draws x 3 langs; draws SHARED across languages (one seeded RNG) so cross-language differences are not draw noise |
| normabl | 9 | K in {48,24,10} x 3 langs, `normalization_scope: subset`, pairs with `topk{K}_seed4000` |
| matched | 6 | top-48 U lip contour (88) vs the gate's own top-88, x 3 langs, equal budget |

Shared recipe: `joint_pruning: False` (subset is fixed, no gate), `max_steps: 40000`,
`early_stop: 30`, `n_gpus: 1`, 16 CPUs. Same recipe as the region ablation, so face-only and
lips-only sit on the same scale as the top-K frontier.

### Cascade — `13585411`, array 0-2, one language per task, 32 CPUs
`scripts/run_cascade.py` + `sbatch/run_cascade.sh`. Four gate trainings per language, run
sequentially in one job (270 -> 100 -> 48 -> 24, each selecting the next subset); the top-270
pool is seeded from the existing 543-wide ranking. Ranks within the pool then maps back to
543-space. 32 CPUs rather than the bulk queue's 16 because the chain is sequential and
latency-bound -- only 3 tasks run at once, so halving per-stage wall clock shortens the
critical path instead of costing concurrency. Chain aborts on a failed stage and propagates
the real exit status rather than reporting COMPLETED.

**Still to launch:** cascade stage-2 (5 K x 3 seeds x 3 langs = 45, blocked on the chains) and
TEST passes for everything (~160 configs, 1-GPU, minutes each).

### Cascade attempt 1 failed; attempt 2 (`13585453`) running

`13585411` died in 30 s on all three languages:
`Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`
raised inside the `src/train.py` child. The exit-status propagation worked -- the jobs reported
FAILED rather than COMPLETED, and the chain stopped instead of continuing on stale subsets.

**Hypothesis tested and REJECTED:** that importing numpy/torch in the orchestrator left
`MKL_THREADING_LAYER` in `os.environ` for the child. Measured: importing numpy or torch sets no
MKL variables in this env, and a parent-imports-torch / child-imports-numpy reproduction runs
clean on the login node.

**Fix applied (pattern-match, root cause NOT confirmed):** every job that works invokes training
as `srun python src/train.py`; the cascade driver called `python src/train.py` from a Python
parent. Orchestration moved into bash in `sbatch/run_cascade.sh`, matching the proven invocation.
`scripts/run_cascade.py` gained `--stages` / `--seed-pool` / `--emit-config` so it only writes
configs; `scripts/extract_gate_ranking.py` reads checkpoints in its own process. Python never
spawns training now.

**Selection step validated independently:** `extract_gate_ranking.py` run against the AUTSL
543-wide checkpoint reproduces `select_top_k_joints.py`'s top-100 exactly -- identical 100
indices, empty symmetric difference.

Attempt 2 is training on all three languages. AUTSL cascade runs at **11.95 it/s on 32 CPUs**
against 5.85 it/s for the 16-CPU stage-2 jobs (2.04x), matching the probe scaling and confirming
the 32-CPU choice for the latency-bound chain.

## Signer composition and the val-test gap — 2026-09-06

**CORRECTION.** An earlier note in this session claimed GSL has "15 signers shared across all
splits". That was a bad regex matching the scene prefix (`police1`, `health1`), not the signer
field. The truth, from `data/gsl/*.csv` filenames:

| dataset | train signers | val signers | test signers | source |
|---|---|---|---|---|
| AUTSL | 31 | 6 | 6 | filename `signerN`, all three splits disjoint |
| ASL Citizen | 35 | 6 | 11 | official `ASL_Citizen/splits/*.csv` `Participant ID`, all disjoint |
| GSL | 6 (1,2,4,5,6,7) | **1 (signer3)** | **1 (signer3, same person)** | filename `signerN` |

**GSL val and test are the same single signer.** Every GSL number in the paper is single-signer
evaluation. This must be stated as a limitation.

**Per-signer test accuracy (top-48, old run set, from the retained per-clip predictions):**

| dataset | n test signers | mean | sd | spread |
|---|---|---|---|---|
| AUTSL | 6 | 86.13 | 3.81 | 9.3 pp (80.66-89.98) |
| ASL Citizen | 11 | 51.19 | **10.47** | **33.7 pp (35.12-68.85)** |
| GSL | 1 | 70.54 | n/a | n/a |

ASL Citizen per signer: P47 68.85, P42 64.25, P35 60.99, P18 56.60, P48 55.22, P49 49.93,
P15 49.00, P9 43.01, P22 41.27, P17 38.90, P6 35.12.

**This explains the val-test gap** (old run set, all 9 runs per dataset):
AUTSL -0.32 to -1.40 · GSL +0.36 to +3.53 · ASL Citizen **-10.70 to -12.76**.

All three are signer-disjoint train-vs-eval, so the gap is not leakage and not overfitting. It is
signer-sampling variance: ASL Citizen's per-signer sd is 2.7x AUTSL's, and with 6 val signers vs
11 *different* test signers the val signers simply happen to be easier. An ~11 pp gap is about
2 standard errors of the difference of means.

**Do not try to "close" this gap** -- it is the honest cost of the official signer-independent
protocol working as designed. Instead: report test as primary for ASL Citizen with per-signer
mean +/- sd (a pooled number hides a 33.7 pp range), note that val is a high-variance proxy there
so val-selected checkpoints are optimistically biased, and state GSL's single-signer limitation.
This is the signer-generalization analysis reviewer pa2X asked for (W4).

## TEST results, new run set — jobs 13594080 / 13594081, 27/27 COMPLETED

Learned top-K, 3 seeds, 1-task `test_visfilt_array.sh`.

| dataset | K | val | test | gap |
|---|---|---|---|---|
| AUTSL | 543 | 84.69 ±0.58 | 83.40 ±1.16 | -1.29 |
| AUTSL | 48 | 89.79 ±0.82 | **89.50 ±1.34** | -0.29 |
| AUTSL | 10 | 84.52 ±0.62 | 84.47 ±0.50 | -0.04 |
| ASL Citizen | 543 | 67.86 ±0.17 | 55.33 ±0.48 | -12.53 |
| ASL Citizen | 48 | 72.00 ±0.43 | **61.79 ±0.30** | -10.21 |
| ASL Citizen | 10 | 62.84 ±0.45 | 51.74 ±1.04 | -11.10 |
| GSL | 543 | 65.53 ±0.61 | 68.82 ±1.40 | +3.29 |
| GSL | 48 | 68.21 ±1.85 | **70.39 ±1.77** | +2.18 |
| GSL | 10 | 64.71 ±1.23 | 67.30 ±1.34 | +2.60 |

**The core claim holds on TEST, not just validation.** K=48 vs the full skeleton:
**+6.09 (AUTSL), +6.46 (ASL Citizen), +1.57 (GSL)**. At K=10 the picture splits: +1.07 AUTSL,
-1.51 GSL, -3.59 ASL Citizen.

ASL Citizen test at K=48 is **61.79** against the submitted paper's 51.18 at the same K -- the new
gate plus recipe is worth ~10 points on held-out signers.

**Val-test gaps reproduce the old run set exactly** (old: AUTSL -0.3 to -1.4, ASL Citizen -10.7 to
-12.8, GSL +0.4 to +3.5), confirming the signer-variance explanation is a property of the splits
rather than of any particular model or recipe.

**Claim I tempered:** I suggested pruning might improve signer generalization because AUTSL's gap
shrinks with K. Across all 27 the effect is weak and non-monotonic -- ASL Citizen's gap is -12.53
at K=543, -10.21 at K=48, but back to -11.10 at K=10. Worth at most a sentence, not a claim.

**Hand asymmetry in the selected subsets** (for the Limitations handedness point, which is
measured rather than assumed):

| dataset | K=48 | K=24 | K=10 |
|---|---|---|---|
| AUTSL | LH 21 / RH 17 | LH 16 / RH 6 | **LH 9 / RH 0** |
| ASL Citizen | LH 17 / RH 21 | LH 6 / RH 12 | LH 3 / RH 6 |
| GSL | LH 18 / RH 21 | LH 2 / RH 15 | **LH 1 / RH 9** |

Balanced at K=48, strongly one-sided at K=10, and the side follows the corpus.

Draft Limitations subsection: `/tmp/limitations_signer.tex` (385 words).

## Statistical tests — 2026-09-06 (CPU only, no GPU)

### Paired significance tests (seed-paired, n=3, two-sided t-test on val)

**Pruned K vs full skeleton (K=543):**

| dataset | significant K | not significant |
|---|---|---|
| AUTSL | 100 (p=.0089), 48 (p=.0188), 24 (p=.0081) | 270 (p=.10), 10 (p=.81) |
| ASL Citizen | 270 (.0139), 100 (.0007), 48 (.0045), 24 (.0034), 10 (.0049, NEGATIVE -5.02) | — |
| GSL | **none** | 270 (.15), 100 (.086), 48 (.084), 24 (.097), 10 (.39) |

**GSL's pruning gains are NOT statistically significant at n=3** — its seed sd (up to 1.85) is
large relative to a ~2.7 pp effect. The paper must say so rather than reporting +2.69 as a result.
95% CIs at K=48: AUTSL [+2.05, +8.15], ASL Citizen [+2.95, +5.33], GSL [-0.90, +6.27].

**Cascade vs top-K at equal K:** significant only at AUTSL K=24 (p=.041) and K=10 (p=.015), and
ASL Citizen K=24 (p=.012). ASL Citizen K=10 is marginal and negative (-1.33, p=.052). GSL: none.
AUTSL K=48 is identically zero — the cascade and direct subsets are the same 48 landmarks.

### Cross-lingual overlap permutation test (20,000 permutations)

| K | observed 3-way | uniform null | p | stratified null | p |
|---|---|---|---|---|---|
| 270 | 213 | 66.81 | <.0001 | 102.81 | <.0001 |
| 100 | 72 | 3.39 | <.0001 | 52.90 | <.0001 |
| 48 | 42 | 0.38 | <.0001 | 32.38 | <.0001 |
| 24 | 8 | 0.05 | <.0001 | 2.97 | .0001 |
| 10 | 1 | 0.00 | .0032 | 0.06 | **.0627** |

The simulated uniform null reproduces the corrected analytic value **K³/543²** exactly (66.81 vs
66.76, 3.39 vs 3.39, 0.38 vs 0.38), confirming that formula — the submitted paper used K⁴/543³,
which was wrong because it counted the multilingual union as a fourth independent language.

The **stratified null** is the honest test: it holds each language's face/pose/hand counts fixed
and shuffles within anatomical groups, so it asks whether languages agree on *specific landmarks*
beyond what shared body-part composition already forces. Agreement survives that at K>=24 but
**not at K=10** (1 observed vs 0.06 expected, p=.063). Claims of cross-lingual convergence should
be made at K>=24 and explicitly qualified at K=10.

## Runs launched 2026-09-06

- `13594261` — TSLFormer baseline TEST, 9 configs (the `_2gpu` variants; the non-2gpu configs were
  never trained). Recipe mismatch investigated and **dismissed**: every TSLFormer run early-stopped
  at 32k-39k steps, well under 40,000, so its 200,000 `max_steps` never bound.
- `13594363` / `13594364` — **ASL Citizen extended-budget re-runs**, 50 configs at
  `max_steps: 80000`, `configs/stage2_ext/`, saving to `models/asl_citizen/stage2_ext/`.
  Reason: 38 of 50 ASL Citizen stage-2 runs reached >=39,000 steps, i.e. were truncated by the
  40,000 cap while still improving, whereas AUTSL and GSL hit 0 of 50. Our ASL Citizen numbers are
  therefore undertrained and understate the method; TSLFormer converged on its own. Internal
  comparisons remain valid (all arms equally truncated) but the external one was skewed against us.

## TEST results for the full matrix — 129 passes, jobs 13594163/4, all COMPLETED

**Caveat: ASL Citizen rows below come from the truncated (40k) runs; `13594363/4` re-runs them at
80k and will supersede them.**

### Cascade vs top-K, now on TEST (3 seeds, paired t-test)

| dataset | K | top-K | cascade | Δ test | Δ val (before) | p |
|---|---|---|---|---|---|---|
| AUTSL | 100 | 86.76 | 86.29 | -0.47 | -0.62 | .700 |
| AUTSL | 48 | 89.50 | 89.50 | +0.00 | +0.00 | identical subset |
| AUTSL | 24 | 87.97 | 88.52 | +0.54 | +0.82 | .483 |
| AUTSL | 10 | 84.47 | 87.44 | **+2.97** | +3.38 | **.028** |
| ASL Citizen | 100 | 60.62 | 60.86 | +0.24 | -0.21 | .090 (sign flipped) |
| ASL Citizen | 48 | 61.79 | 62.38 | **+0.59** | +0.35 | **.019** |
| ASL Citizen | 24 | 59.29 | 60.89 | **+1.60** | +1.13 | **.008** |
| ASL Citizen | 10 | 51.74 | 52.26 | **+0.52** | **-1.33** | .563 (SIGN FLIPPED) |
| GSL | 100 | 71.43 | 70.25 | -1.18 | -0.63 | .286 |
| GSL | 48 | 70.39 | 70.49 | +0.10 | -0.42 | .954 |
| GSL | 24 | 70.81 | 71.75 | +0.94 | +1.24 | .557 |
| GSL | 10 | 67.30 | 69.30 | +1.99 | +1.86 | .052 |

**Three corrections to what was reported from validation:**
1. **The ASL Citizen K=10 "honest exception" is gone.** On val the cascade lost 1.33; on test it
   gains 0.52 (ns). The error-propagation story told about that point does not survive.
2. **AUTSL K=24 is no longer significant** (val p=.041 -> test p=.483).
3. **ASL Citizen K=48 becomes significant** on test (p=.019) where it was not on val.
Net: the cascade case is *stronger* on test — significant at ASL Citizen K=48 and K=24 and
AUTSL K=10, with GSL K=10 marginal (p=.052).

### Normalization ablation on TEST (n=1; replicates in flight)

Direction matches val everywhere and is larger: AUTSL -5.37 / +0.40 / -6.36, ASL Citizen
-0.80 / -2.11 / -2.71, GSL +2.34 / +3.29 / +2.29 at K=48/24/10. Subset-statistics normalization
helps GSL and hurts the other two. Still n=1.

### Matched budget on TEST (n=1) — now essentially NULL

top-48 U lips vs gate top-88: AUTSL **-0.08**, ASL Citizen **+0.43**, GSL **+0.31**.
On val these were +0.96 / +1.58 / -1.35. On test adding the lip contour at equal budget makes no
difference at all. This strengthens the redundancy reading: the mouth information is already
recoverable from what the gate selects. Do not claim the gate leaves usable information on the
table.

### Region ablation on TEST (n=1)

| dataset | face-only | lips-only | chance | face x chance |
|---|---|---|---|---|
| AUTSL | 13.87 | 16.30 | 0.44 | 31x |
| ASL Citizen | 1.07 | 1.40 | 0.04 | 29x |
| GSL | 9.29 | 10.66 | 0.32 | 29x |

Both conclusions hold on test: the face carries real signal (29-31x chance on all three), and
**40 lip landmarks still beat the whole 468-point mesh** on all three. The Limitations claim
stands.

## Paper rewrite pass — 2026-09-07

Backup: `docs/.backups/emnlp_v2.<ts>.pre-method.tex`. Canonical numbers in
`docs/RESULTS_SNAPSHOT.txt` (regenerate after job 13603700/13603701 lands).

**Rewritten to match what we actually run:**
- **Method §3.2** — replaced the annealed-sigmoid description (p0=0.98, tau 10->0.01, lambda_max=20,
  warmup 10k) with the hard-concrete formulation: stretch-and-clamp gate, (gamma,zeta)=(-0.1,1.1),
  beta=2/3, closed-form open probability, penalty normalized by J only, p0=0.5 / lambda=2 / Ta=5000.
  States the exact-zero counts (309/250/163).
- **Abstract** — all `\prelim` removed (0 remaining). The old "+84 points at K=10" claim was not
  supported; replaced with measured 16-36 pp at K=48 and 36-53 pp at K=10, plus the TSLFormer margins.
- **Results §5.1-5.2** — rewritten on test numbers with paired t-tests. Corrects the submitted paper's
  "K=10 matches or beats the full skeleton on three of four datasets": at K=10 only AUTSL is above
  baseline (+1.07, ns), ASL Citizen is -6.02 and GSL -1.82. Cascade claim narrowed to K=10.
- **Overlap §5.3** — old four-dataset consensus counts replaced with three-language counts and the
  permutation test; corrected null is K^3/543^2. Adds the stratified null and restricts the
  convergence claim to K>=24 (K=10 is p=.063 against it). Fixes reviewer pa2X's C3: non-manuals
  carry lexical as well as grammatical contrasts.
- **Conclusion / Limitations** — Limitations rebuilt with four paragraphs: signer generalization
  (per-signer spreads, the ASL Citizen val-test gap as sampling, GSL's single signer), the face
  (region ablation + the per-group standardization control), coverage/encoder-agnosticism (including
  the attention confound we cannot rule out), and vocabulary distribution (long-tail + handedness).
- **Appendix E** — new gate hyperparameter sweep table (ABDG's requested schedule ablation).
- Multilingual residue removed from the dataset table, setup, evaluation, and reproducibility
  checklist. Optimization appendix corrected (1 GPU, 40k/80k steps, stage-1 fixed budget).

All `\ref` targets resolve; no duplicate labels; body prose ~3,720 words.

**Still outstanding:** four figures (1, 2, 3, 5); the preprocessing-transparency paragraph and
detection-rate table in Setup; the leakage/stage-independence statement; 10 missing citations
(Vb1P); compute appendix numbers; code + learned-indices release.

**VENUE UNRESOLVED.** The draft is built against `\usepackage[review]{acl}`. Target is now COLING,
whose style files differ. Could not confirm COLING 2026/2027 requirements from the web; COLING 2025
used 8 pages for long papers with unlimited references and appendix, on COLING style files adapted
from the ACL template. Needs the actual CFP before the format pass.

## Hand-crafted baseline, final — 5 seeds per arm (jobs 13605990 / 13606025 / 13607233)

The 3-seed comparison was underpowered AND the arms were mismatched (TSLFormer had 3 seeds, ours 5),
so pairing discarded two of ours. Added TSLFormer seeds 4003/4004 for all three languages.

| dataset | TSLFormer | ours K=48 | delta | paired p | 95% CI |
|---|---|---|---|---|---|
| AUTSL | 87.04 ±0.37 | 88.71 ±1.44 | +1.67 | .054 | [-0.04, +3.38] |
| ASL Citizen | 60.96 ±1.21 | 63.32 ±0.80 | +2.36 | **.049** | [+0.01, +4.70] |
| GSL | 66.85 ±1.27 | 70.43 ±2.03 | +3.58 | **.014** | [+1.18, +5.99] |

**Correction history on this comparison, worth remembering.** First reported as "significant on all
three (p=.041/.019/.019)" -- that used an UNPAIRED EQUAL-VARIANCE t-test on 3 seeds, the least
conservative option available, and was wrong. Paired at 3 seeds it was .087/.105/.074 (nothing
significant). Paired at 5 seeds per arm it is .054/.049/.014. Two of three now fall just inside the
threshold and one just outside, so the paper reports effect sizes and CIs rather than leaning on
significance verdicts.

Note the margin ordering is interpretable: TSLFormer's 50 landmarks were hand-picked for Turkish
Sign Language, and the gain over it is smallest on AUTSL (Turkish) and largest on GSL.

**All 335 trained runs now have test results.** Straggler lesson, hit four times: array counts
reporting COMPLETED never implies test coverage when training and testing overlap. The only
reliable gate is enumerating every config and checking for `checkpoints/` without
`predictions/*.metrics.json`.

## Per-anatomical-group detection rates — job 13680832, 18,000 clips (2,000 per split x 3 x 3)

`scripts/measure_detection_rates.py` -> `data/detection_rates.json`. MediaPipe Holistic returns
NaN for a whole component when its detector does not fire; components are never partially present,
so a frame-level rate per group is well defined. "after fill" = after our linear gap filling up to
5 consecutive frames, i.e. what the encoder actually receives.

| dataset | face raw | body raw | L hand raw | R hand raw | L hand filled | R hand filled | body in-image |
|---|---|---|---|---|---|---|---|
| AUTSL | 99.74 | 100.00 | 86.24 | 91.13 | 99.60 | 98.91 | 80.37 |
| ASL Citizen | 99.78 | 99.96 | **29.88** | **37.33** | **42.63** | **51.52** | 52.64 |
| GSL | 99.98 | 100.00 | 93.28 | 92.68 | 99.34 | 99.46 | 75.71 |

**Answers reviewer ABDG's hypothesis in the opposite direction.** They asked whether the gate drops
the face because of systematic detection failure. The face is the MOST reliably detected region on
every corpus (99.7-100%), and the hands are the LEAST -- yet the gate selects hands most heavily and
the face not at all. Detection reliability is not what drives selection.

**Unanticipated and important: MediaPipe misses the hands in most ASL Citizen frames.** Raw hand
detection is 30% (left) / 37% (right); gap filling only lifts it to 43% / 52%, because the
dropouts are long runs rather than short flickers. ASL Citizen is crowdsourced webcam video recorded
in participants' homes. This is a plausible driver of (a) ASL Citizen's much lower absolute accuracy
(57-63% vs 83-89% AUTSL), and (b) the 33.7 pp per-signer spread, if hand detection varies by
recording setup. The latter is directly testable: correlate per-signer hand detection with
per-signer accuracy.

**Body in-image rate quantifies the phantom-landmark issue:** only 52.6% of body landmarks lie inside
the frame on ASL Citizen (75.7% GSL, 80.4% AUTSL); the remainder are regressed by BlazePose for
body parts outside the camera view.

## Self-review pass (simulated ARR reviewer) — 2026-09-14

Backup: `docs/.backups/emnlp_v2.*.pre-review-fixes.tex`.

### Most important finding: the learned subset is nearly the hand-crafted one
The TSLFormer subset is all 42 hand landmarks + 8 upper-body (no face). Our learned K=48 subset
shares **42/48** with it on all three languages (composition: 1-2 more body, 3-4 fewer hand
landmarks). The method largely rediscovers the practitioner heuristic. Paper now reframed around
this: recovering the manual choice from the objective is presented as validation, and the gain over
the full skeleton is not oversold as the learned ranking's own contribution.

### Verified problems found and fixed
- Conclusion still said "the L0 gate does not converge to a binary {0,1} regime" -- contradicted
  Method (309/250/163 exact zeros), and was the sentence reviewer ABDG quoted. Removed.
- App. Training still said "All reported results use a single random seed (seed = 4000)" -- the
  other sentence ABDG quoted. Now describes 5 seeds. Hyperparameters were the OLD gate's (p0=0.98,
  lambda_max=20, tau annealing); replaced with hard-concrete values.
- Baselines said "single random draw at each K in {270,...,10}" (false) and listed only two
  baselines (TSLFormer omitted). Now four baselines, correct draws.
- Cascade: "no p-value falls below .28" was FALSE (GSL K=100 p=.082). Corrected.
- Significance: "all fifteen contrasts in this range positive" -- the K=24-100 range is nine
  contrasts. Corrected.
- MY ERROR RETRACTED: I had written that TSLFormer's subset was "selected for Turkish SL" and that
  this explained the margin ordering. The subset is generic and overlaps 42/48 on every language.
- Related Work called the gate "sigmoid-based"; Method 3.1 kept ABDG's "invariant to which
  landmarks" phrasing; Objective used tau notation and "eight GPUs"; stage 2 described "with the
  gate held fixed" (it has no gate); "four corpora"; `\cite{rgb}` placeholder (removed -- add a
  real RGB-ISLR citation if wanted).
- Mechanism contradiction (W6): contributions/conclusion claimed the criterion "tracks displacement
  magnitude" while Limitations reported the amplitude-equalization control did not change
  selection. Amplitude claim removed; redundancy stated as an untested hypothesis.
- Unsupported generalization about label-space granularity removed.

### Statistical corrections
- **Holm correction over the 15 pruning contrasts:** only ASL Citizen K=48 (adj p=.002) and AUTSL
  K=24 (adj p=.010) survive; headline AUTSL K=48 goes .005 -> .058. Paper now claims a consistent
  direction (all 9 contrasts in K=24-100 positive) supported by two surviving contrasts.
- **Uniform random-K is a weak control at small K:** hands are 42/543, so a K=10 draw has 0.77 hand
  landmarks in expectation and none 44% of the time (14% at K=24). Launched a composition-matched
  control (below).

### Other additions
- W7: cited differentiable feature selection -- `yamada2020stg`, `abid2019concrete` (**must be
  added to custom.bib**). Contribution reframed as an application, not a new mechanism.
- W8: removed the "lighter to train" efficiency implication.
- W5: Limitations now notes GSL checkpoints are selected on data from the test signer.
- SAM-SLR substitution justified in text: its 27 nodes are defined over HRNet's 133 keypoints.

### Runs launched
- **`13682362` -- stage-1 ranking stability**, 9 runs: seeds 4001, 4002 + a seed-4000 rerun per
  language, on the original 2-GPU config so seed effects are not confounded with GPU count. The
  rerun gives the same-seed nondeterminism floor. Configs `configs/gate_stage1_seeds/`. Decide next
  steps after seeing it (user: "look at the seed stability and then decide").
- **`13682377/13682378` -- composition-matched random baseline**, 45 runs: 5 draws x K{48,24,10} x
  3 langs, each matching the learned top-K subset's exact face/body/LH/RH counts and randomizing
  within groups. Isolates within-part landmark choice from which parts are kept (the W1 question).
  Index files `data/region_subsets/matched_random/`.

Body prose is now ~4,119 words (up from ~3,934): the reframing and Holm paragraph added length.

## Stage-1 ranking stability — jobs 13682362 + 13691581, 9 runs, all COMPLETED

Answers the strongest self-review criticism (W2): every stage-2 result rested on a single stage-1
ranking per language, so the method's actual product had no variance estimate. Design: seeds 4001
and 4002 per language, plus a **rerun of seed 4000** to give the same-seed nondeterminism floor.
All on the original 2-GPU config so seed effects are not confounded with GPU count.

| | same seed (n=3) | different seeds (n=15) |
|---|---|---|
| Spearman rho | 0.995 [0.987, 1.000] | 0.901 [0.870, 0.921] |
| top-100 overlap | 97.0/100 | 81.5/100 (77-86) |
| **top-48 overlap** | **48.0/48** | **47.2/48 (46-48)** |
| top-24 overlap | 23.7/24 | 22.5/24 (22-24) |
| top-10 overlap | 10.0/10 | 9.0/10 (7-10) |

**Verdict: no further work needed.** The K=48 subset behind the headline results is reproduced to
within two landmarks across every cross-seed pair, so stage-2 need not be rerun against alternative
rankings. Written up as Appendix "Stage-1 Ranking Stability" with a pointer from Limitations.

The instability that does exist sits at K=10 (7-10/10) and K=100 (77-86/100). The K=10 figure
coheres with three independent findings: pruning stops helping there, the cascade gains most there,
and cross-language agreement fails the stratified null there. All four are the same fact about the
tail of a 543-wide ranking being under-determined.

Note ASL Citizen and GSL reproduced their seed-4000 ranking essentially exactly on rerun
(rho 0.999 / 1.000) while AUTSL did not (0.987), so run-to-run nondeterminism is node-dependent.

## Black-hole node dw-2-4 — 2026-09-14

25 of 54 jobs across `13682362` and `13682377` FAILED in 30-106 s, **every one on dw-2-4**, at
`torch._C._cuda_init()` inside Lightning's `_check_cuda_matmul_precision`. Every other node
succeeded. A node that fails fast frees up fast, so the scheduler kept feeding it work.
`docs/submission_instructions.md` already listed dw-2-4 and cs-1-2 as known-bad; that was missed.

The crashed runs left 25 empty save directories (0 KB, no checkpoints), verified empty and removed
with bottom-up `os.rmdir`, which refuses non-empty directories. All 25 resubmitted as
`13691581`-`13691583` with `--exclude=dw-2-4,cs-1-2` and completed.

`--exclude=dw-2-4,cs-1-2` is now an `#SBATCH` directive in `train_visfilt_1gpu.sh`,
`train_visfilt_2gpu.sh`, `test_visfilt_array.sh`, `run_cascade.sh` and
`measure_detection_rates.sh`, so it no longer depends on remembering a CLI flag.

## Composition-matched random baseline — 45 runs trained, tests `13721344`/`13721345` in flight
