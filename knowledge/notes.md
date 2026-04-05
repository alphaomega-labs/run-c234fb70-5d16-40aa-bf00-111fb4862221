# Knowledge Notes: CCGO-RL

## Coverage Snapshot
- Total sources: 43
- Primary papers/reports/articles: 40
- Recent sources (2023+): 13
- Corpus reconciled to one canonical record per URL across refs/index/payload.

## Goal-Conditioned Control Lineage
- UVFA formalized goal-conditioned value approximation and shared generalization across goals [src_eb6929ff0d].
- HER made sparse-goal off-policy replay practical by relabeling achieved goals [src_98166b0b87].
- GCSL/contrastive families move from binary relabeling to representation-aware goal scoring [src_2ab0eb48b4, src_bfba3bebd4].
- Recent 2026 goal-space methods (ViSA, ACDC, GraSP-STL) emphasize augmentation/curriculum/planning overlays but still rely on replay-conditioned goal-value structure [src_3bd2765fcc, src_fefda046d2, src_62c8df7c8d].

## Intrinsic Motivation and Exploration Thread
- Count-based pseudo-count ideas establish principled novelty bonuses in high-dimensional spaces [src_8f85632d6f].
- Prediction-error curiosity (ICM/RND lineage) provides scalable intrinsic signals with known non-stationarity risks [src_d8e5b84eba, src_f0fb14bdcd].
- Directed exploration frameworks (NGU/Agent57) combine episodic and lifelong novelty components for deceptive sparse rewards [src_073a4abc44, src_17eb5b0439].
- Entropy/state-coverage and skill-discovery methods improve broader state visitation and pretraining transfer [src_fc279bb42b, src_138b68a6f7, src_8125133624].

## Objective and Equation Seeds for CCGO-RL
- Dual-stream return model: `J = E[sum_t gamma^t (r_t^e + beta_t r_t^i)]` with annealed `beta_t` tied to goal confidence.
- Goal-conditioned Bellman update seed: `Q(s,a,g) <- r^g(s,a,s') + gamma max_{a'} Q(s',a',g)` from UVFA/HER lineage.
- Contrastive goal value proxy seed from CRL: `Q(s,a,g) ~ log p(g|s,a) - log p(g)`.
- Skill/novelty intrinsic form from DIAYN: `r_z(s,a) = log q_phi(z|s) - log p(z)` as reusable curiosity primitive.
- UNREAL-style stability template: `L_total = L_base + sum_k lambda_k L_aux,k` for bounded auxiliary influence.

## Assumptions Reused Across Papers
- Replay buffers provide sufficiently diverse transitions for both extrinsic and intrinsic/value updates.
- Goal representation is either explicit in observations or recoverable through embeddings.
- Non-stationary intrinsic signals can be controlled by scheduler/normalization without collapsing final task return.
- Off-policy actor-critic stability relies on conservative target updates, reward scaling discipline, and seed-averaged evaluation.

## Benchmark Protocol Signals
- Environment substrate: Gymnasium + DM Control + MiniGrid + Procgen + D4RL [src_18ed07dbc3, src_0fdbd47b59, src_c5c1337248, src_935b504ec6, src_fa51915d26].
- Core metrics for this project context: final return, area-under-learning-curve, first-success time, coverage, and stability under intrinsic annealing.
- Minimum reporting discipline: multi-seed means + confidence intervals + ablations over coupling/scheduler variants.

## Comparator Lineage for Future Distillation
- Goal-conditioned baselines: UVFA, HER, GCSL, contrastive RL variants.
- Intrinsic baselines: ICM, RND, pseudo-count, RE3/APT-style entropy/state-coverage objectives.
- Optimization baselines: SAC, TD3, DDPG, PPO depending on discrete/continuous control and compute envelope.
- Hybrid references for decomposition design: Agent57, Plan2Explore, UNREAL.

## Major Similarities and Contrasts
- Similarity: strongest methods separate representation/novelty estimation from policy optimization but share replay and target-network machinery.
- Difference: novelty definition (prediction error vs count/entropy vs skill MI) changes stability properties and exploration bias.
- Difference: scheduler design (fixed coefficient, adaptive correlation, confidence-based annealing) largely determines asymptotic-return preservation.
- Difference: evaluation rigor varies; many preprints under-report cross-benchmark seed robustness and reward-hacking diagnostics.

## Limitations and Open Gaps for Next Phases
- Compute budget is still TBD, so benchmark breadth/depth tradeoffs remain unresolved for CPU-only execution.
- Several newest sources are arXiv preprints and require venue-version cross-check before manuscript claims are finalized.
- Equation extraction remains uneven for some PDFs due encoding artifacts; fallback is TeX/source-level parsing in literature_review.
- Novelty-hacking safeguards and non-stationary reward normalization design need formalization in hypothesis_methodology.

## Retry-4 Additions
- Added Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play (ICLR 2018; doi:10.48550/arXiv.1703.05407): automatic curriculum generation via Alice/Bob self-play for sparse-reward transfer.
- Full-text fetched and extracted under `knowledge/raw/pdfs/1703.05407.pdf` and `knowledge/raw/pdftxt/1703.05407.txt`.
- This addition strengthens coverage of intrinsic curriculum mechanisms complementary to novelty bonuses and goal-conditioning.

