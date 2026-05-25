#!/usr/bin/env bash
# 双卡环境分阶段部署。同时适配 5090(Blackwell sm_120)和 4090(Ada sm_89)。
#
# 设计原则:只 pin 不让步的版本(torch 2.7+cu128 整套、vllm ≥ 0.9、trl ≥ 0.11),
# 其余包交给 pip 反向约束自动解析。装错版本会直接跑不起来的才 pin。
#
# 关键陷阱:vllm 0.8.x 整支硬钉 torch==2.6.0(无 sm_120),所以为兼容 Blackwell 必须 vllm ≥ 0.9。
# cu128 wheel 同时包含 sm_89 / sm_120 kernels,4090 直接复用同一套版本号即可,
# 不必为了 Ada 单独退回 cu121 —— 那样 trl/transformers/vllm 的版本约束都要重新验。
# 同时 torchvision/torchaudio 要和 torch 同源(同一 cu128 wheel),否则 transformers
# lazy-import torchvision 时会因为 ABI 不匹配在 torchvision::nms 上炸。
#
# 用法(4090 主机,落在 NVMe 数据盘):
#   ENV_PREFIX=/root/shared-nvme/conda/envs/sketch4090 bash v2/scripts/setup_env.sh
#   conda activate /root/shared-nvme/conda/envs/sketch4090
# 用法(5090 主机,沿用旧默认):
#   ENV_PREFIX=/data/conda/envs/sketch5090 bash v2/scripts/setup_env.sh
#
# 出错时:本脚本设了 -e,某一步失败会立刻退出,可从那一步往下手动重跑。

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/root/shared-nvme/conda/envs/sketch4090}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "[0/5] conda create (python=3.10) → $ENV_PREFIX"
conda create -p "$ENV_PREFIX" python=3.10 pip -y

# 后续步骤都在新环境里跑;不用 `conda activate`(脚本里激活不稳),直接走 prefix 的 pip。
PIP="$ENV_PREFIX/bin/pip"

echo "[1/5] torch + torchvision 2.7 / cu128 (Blackwell sm_120 必须;cu128 wheel 也含 sm_89 内核)"
# torch + torchvision 一起装,保证 ABI 同步:transformers lazy-import torchvision,不同步会在 nms 上炸。
# 不装 torchaudio:PyTorch cu128 索引不发布带 +cu128 后缀的 torchaudio(它没那么强的 CUDA 耦合),
# 我们整个 v2 代码库也没有任何地方 import torchaudio,留着只会引入版本协商麻烦。
"$PIP" install --index-url "$TORCH_INDEX" \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128

echo "    verify: torch.cuda + (sm_89 Ada 或 sm_120 Blackwell)"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability()
assert cap in {(8, 9), (12, 0)}, f"expected sm_89 (Ada/4090) or sm_120 (Blackwell/5090), got sm_{cap[0]}{cap[1]}"
print(f"  torch={torch.__version__}, device={torch.cuda.get_device_name()}, cap={cap}")
PY

echo "[2/5] vllm >=0.9,<0.11 (按 torch 2.7 编译的支线;Blackwell 必须 ≥0.9,Ada 同样能用)"
# 不加 --no-deps:vllm 运行时依赖很多,让 pip 解析。
# 用范围而非精确版,留出 patch release 升级空间;装完回头核验 torch 没被换 + pip check。
"$PIP" install "vllm>=0.9,<0.11"

echo "    verify: torch still cu128 and pip tree consistent"
"$ENV_PREFIX/bin/python" - <<'PY'
import torch
assert "+cu128" in torch.__version__, f"torch was downgraded: {torch.__version__}"
print(f"  torch={torch.__version__} OK")
import vllm
print(f"  vllm={vllm.__version__}")
PY
# pip check 把"X requires torch==Y"这类隐性回退一次性查出来;不通过就停。
"$PIP" check

echo "[3/5] 训练栈 (trl 0.11-0.15 是硬约束,其余跟它解析)"
# transformers 必须 <5:5.x 移除了 all_special_tokens_extended 等内部属性,
# 同时 AutoTokenizer 的 fast/slow 选择行为也变了,vllm 0.10.x 在 init 时直接 AttributeError。
# vllm 适配 transformers 5 之前都得卡在 4.x。
#
# trl 必须 <0.16:0.16 起 DataCollatorForCompletionOnlyLM 被移除,1.x 大改 SFTTrainer/DPOTrainer
# API surface。我们的 DynamicSFTCollator 继承自这个类,不降级无法 import。
# 等以后整体重写 collator 走 SFTConfig(completion_only_loss=True) 再松上界。
"$PIP" install "trl>=0.11,<0.16" "transformers<5" accelerate peft deepspeed datasets

echo "    verify: trl 关键导入(包括 0.16+ 移除的 DataCollatorForCompletionOnlyLM)"
"$ENV_PREFIX/bin/python" - <<'PY'
# 这三个一起 import 才能把"trl 太新"的版本溢出 ImportError 在装环境时就暴露,
# 避免拖到训练阶段才发现。
from trl import SFTConfig, DPOConfig, DataCollatorForCompletionOnlyLM
import trl, transformers, accelerate, peft, deepspeed, datasets
print(f"  trl={trl.__version__} transformers={transformers.__version__} "
      f"accelerate={accelerate.__version__} peft={peft.__version__} "
      f"deepspeed={deepspeed.__version__} datasets={datasets.__version__}")
PY

echo "[4/5] 工具 (全不 pin)"
"$PIP" install modelscope pynvml sentencepiece pyyaml tensorboard rich pytest

echo "[5/5] done. 激活环境:"
echo "  conda activate $ENV_PREFIX"
