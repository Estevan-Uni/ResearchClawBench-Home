# EvoLab 在 ResearchClawBench 的跑榜结果

本目录整理 **EvoLab 的 ResearchClawBench G1/G2 40 题结果**。Agent 使用 Claude Code 与 Claude Opus 5.5 high 作答，已完成题目按 G1/G2 较高分选轮。**37 题有 Gemini 4 Carbon judge 评分；`Chemistry_000`、`Chemistry_002`、`Life_001` 的 Agent 最后尝试被安全策略拦截，均按 0 分记录。**这 3 个 0 分是整理时指定的结果，未经 judge 评分；记录仍保留 `status=failed`、原始失败证据和费用/时间，不会冒充正常评出的 0 分。

40 题总表是 [`review_set/all_results.csv`](review_set/all_results.csv)。`score_source` 区分 `gemini_4_carbon_judge` 与 `assigned_zero_for_agent_safety_refusal`，`judge_evaluated` 同时标明是否实际经过 judge。37 题评分细账在 [`review_set/audit.csv`](review_set/audit.csv)，3 题失败细账在 [`review_set/failed_runs.csv`](review_set/failed_runs.csv)。`review_set/runs_index.json` 有 40 条 Home 风格索引，`review_set/failed_runs_index.json` 单列 3 条失败题。没有改动 fork 已发布的 `data/` 排行榜。

## 每题结果目录

`review_set/runs/<TaskID>_<YYYYMMDD_HHMMSS>/` 每题恰好一个目录：

| 文件 | 内容 |
|---|---|
| `data.json` | Home 风格题目结果、报告、评分、模型、费用和时间。 |
| `_meta.json`、`_score.json` | 运行状态和评分。安全拒绝题的 `_score.json` 明确写出人工指定 0 分及 `judge_evaluated=false`。 |
| `source_score.json` | 仅 37 道 judge 评分题有此文件，保存原始评分响应。 |
| `files.json`、`workspace/` | 已复制报告、图片、代码及可浏览的小型输出；`files.json` 列出实际文件。 |
| `output.json`、`_agent_output.jsonl`、`_openevo_transcript.json` | Agent 输出及其记录（有原件时保存）。 |
| `accounting.json`、`attempts.json` | G1、E1、G2 及重试的费用、活动耗时与逐次尝试来源。失败题只记录发生过的 G1 尝试。 |
| `evidence.json` | 原始路径、SHA-256、已复制文件与省略文件统计。 |

36 题有 G1/G2 双分并取高分，`Chemistry_001` 只有 G1 有效分；最终是 **G1 入选 4 题、G2 入选 33 题、安全拒绝记 0 分 3 题**。这是 **best-of-two evolutionary gate**，不能称为 Pass@1。`Information_000` 的 G2 57.80 高于 G1 43.10，因此选择 G2。原始批量 CSV 对 `Earth_003`、`Information_000`、`Neuroscience_001` 更新滞后，构建脚本改用完成的 attempt 和所有 judge criterion 请求记录。

`duration_seconds` 为已发生 G1/E1/G2 阶段的活动 wall time 之和，包含重试与 judge；`elapsed_including_queue_seconds` 另计首末时间间隔。`known_cost_usd` 汇总已记录的 solver、evolution、judge 请求费用；若有费用未知，`cost_usd=null`，已知数仅为下界。`Astronomy_000` 选 G2 51.60，其全部阶段已知费用 $14.369848、活动耗时 9,880.387 秒、首末耗时 10,430.165 秒。每题 `accounting.json` 和 `attempts.json` 可追溯具体来源。

## GitHub 与官方提交

本目录是公开核对用的 Home 风格归档。为控制仓库体积，复制了报告和代码、可浏览的较小输出，但省略原始 `data/`、`related_work/`、大型模型权重和数组。维护者在 [官方 issue #13](https://github.com/InternScience/ResearchClawBench/issues/13)要求将 **40 题完整运行目录打成单个 ZIP**，其中每题的 `data/` 可省略，再邮件发送给维护者审核。因此**本 GitHub 归档及其 0 分口径尚不能视为官方接受或正式排名**；详见 [提交符合性核对](SUBMISSION_STATUS.md)。

原始实验文件位于仓库外，构建脚本只读。原始 benchmark 源码提交为 `c19a320d3d3d40e370e59001cc219794c4261055`，Home 基线为 `7adb4eef696a0aa15b6668a6e73203235f69b34a`。

## 重建与验证

在仓库根目录运行：

```bash
python3 submission/evolab/build.py
python3 submission/evolab/verify.py
```

`build.py` 不会覆盖已有 `review_set/`。需要重建时，仅删除这个生成目录后再运行。`verify.py` 会对 40 个目录、评分来源、零分标记、费用/时间和源文件哈希进行核对。
