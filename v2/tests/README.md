# v2 Smoke Tests

离线 / 无 GPU / 无 API 的快速验证。所有测试应该在几秒内跑完。

## 运行

```bash
# 从项目根目录(包含 v2/ 的那一层)
pytest v2/tests -v
```

或单文件:
```bash
pytest v2/tests/test_parser.py -v
```

## 覆盖

| 测试文件 | 模块 | 重点 |
|---|---|---|
| `test_prompts.py` | data/prompts | 模板组装顺序、response template 位置 |
| `test_apps_loader.py` | data/apps_loader | difficulty 过滤、io_format 判定、9:1 拆分 |
| `test_parser.py` | sampling/parser | sketch / code 抽取、模型只写一半的容错 |
| `test_compare.py` | execution/compare | stdio 浮点容差、fncall APPS list 剥皮 |
| `test_timing.py` | execution/timing | CoV 收敛 + 上限保护 |
| `test_runner.py` | execution/runner | 真跑 fncall + stdio,partial credit,超时(Linux only) |
| `test_rubric_json.py` | annotation/rubric | judge 输出 JSON 抽取 + clamp |
| `test_pair_builder.py` | training/pair_builder | PvF/QvS/GvB/all 四类对的构造与边界 |
| `test_sft_dataset.py` | training/sft_dataset | top-p% 选择,动态 vs 静态 |
| `test_dpo_dataset.py` | training/dpo_dataset | set_transform 动态产 triple,task 可过滤 |
| `test_merge.py` | merge/build_dataset | parsed_ok 过滤、score 缺失 → -1 |
| `test_metrics.py` | evaluation/metrics | pass@k Codex 公式 + 多题平均 |

## **没**覆盖到的

需要 GPU / API / 大权重的部分:
- `sampling/backend.py` — vllm/HF 真后端
- `annotation/judge_client.py` — GLM-4-Air HTTP 调用
- `training/sft_train.py` / `dpo_train.py` — trl Trainer 端到端
- `evaluation/eval_sampling.py` — 真模型采样

这些需要在 V100 服务器 + ZhipuAI API key 准备好后单独 sanity 一遍。
