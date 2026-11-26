# EvoLab ResearchClawBench 提交符合性核对

核对日期：2026-11-26 UTC。**本地待推送的归档已覆盖全部 40 题；官方邮件提交尚未完成，也未获维护者接受。**

## 官方渠道

[ResearchClawBench 维护者在 issue #13 的答复](https://github.com/InternScience/ResearchClawBench/issues/13)要求将全部 40 题的运行目录放在**同一个 ZIP** 中，每个目录形如 `TaskID_YYYYMMDD_HHMMSS/`。每题的 `data/` 可省略。将 ZIP 发给 `xu_wanghan@sjtu.edu.cn` 后，由维护者审核并决定是否公布。其回复没有说明安全拒绝如何计分、是否接受 G1/G2 取高分、judge 型号和费用口径。GitHub fork 的提交不是上述官方交付渠道。

## 已整理的结果

| 项目 | 状态 |
|---|---|
| 题目覆盖 | 40 题各一目录；`runs_index.json` 与 `all_results.csv` 各 40 条。 |
| Judge 评分 | 37 题；G1 入选 4 题，G2 入选 33 题。36 题有双轮分数按较高分选取；`Chemistry_001` 仅 G1 有效分。 |
| 安全拒绝 | `Chemistry_000`、`Chemistry_002`、`Life_001` 各记 **0 分**，`score_source=assigned_zero_for_agent_safety_refusal`，`judge_evaluated=false`，运行状态保留 `failed`。`Chemistry_002` 的部分报告留存。 |
| 来源 | 37 题保留原始 judge 分数和逐项 rubric；3 题保留失败尝试与人工 0 分标记。 |
| 费用和时间 | 每题记录已发生阶段及重试；未知费用时 `cost_usd=null`，`known_cost_usd` 为下界。 |
| 静态结果 | Home 风格 `data.json`、`files.json`、`output.json`、报告、代码、图片及小型输出已整理，源路径和哈希可核对。 |

## 与官方 ZIP 的差距

本 GitHub 目录是**精简审查归档**，省略了 `related_work/`、大型模型权重和数组及 `data/`。维护者明确允许省略 `data/`，未明确允许省略其他文件。原始 40 题 workspace 去掉 `data/` 后约 **41.8 GB**，其中大型模型权重和数组占多数；当前磁盘可用空间约 17 GB，无法在此工作区生成严格按已知要求保留其余文件的完整 ZIP。当前仓库也没有该 ZIP。

因此，**不能宣称本归档已满足官方提交要求或已经上榜**。正式提交仍需按维护者口径准备完整单 ZIP 并邮件送审；安全拒绝记 0 分、best-of-two 和 judge 选型均由维护者决定是否接受。GitHub 推送仅用于结果公开归档，不代表维护者认可这些口径。
