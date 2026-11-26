# EvoLab ResearchClawBench results

This directory records 40 ResearchClawBench tasks. Opus 5.5 high ran through Claude Code in G1, one task-local evolution step, and G2. Gemini 4 Carbon scored the answers. The gate selects the higher score for each task with two scores and preserves the available G1 score when evolution was blocked.

| Measure | Value |
|---|---:|
| Tasks with two scores | 36 |
| Judge-scored tasks | 37 |
| Unjudged safety refusals counted as zero in the aggregate | 3 |
| Selected G1 / G2 results | 4 / 33 |
| Gate mean across 40 tasks | 45.362 |
| Documented spend (USD) | 1,022.93 |
| Requests or attempts with unavailable cost | 25 |

[Results](review_set/all_results.csv) include score provenance and active time across G1, E1, G2, judging, and retries. [Per-run records](review_set/runs/) contain score and accounting metadata. The 36 report bundles include original report text, figures, and judge responses. Four runs connected to safety blocks include results metadata. Chemistry_001 preserves its judged G1 score of 17.30. Chemistry_000, Chemistry_002, and Life_001 retain failed status and explicit unjudged score markers.

This submission uses best-of-two selection. For Astronomy_000 and Astronomy_002, one rubric item per generation used medium reasoning. Other scored items used high reasoning. [Submission scope](SUBMISSION_STATUS.md) records the review status.
