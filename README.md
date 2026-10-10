# jxmind · MokioMind

一个可在本地完成 **预训练、指令微调（SFT）、检查点恢复和对话推理** 的小型 Decoder-only Transformer 项目。模型包含 RMSNorm、RoPE / YaRN、GQA、SwiGLU 和可选 MoE。

项目名称和安装包统一为 `jxmind`；保留 `MokioMindConfig`、`MokioMindForCausalLM` 和 `model_type=mokiomind`，以兼容已有配置。代码面向学习与实验。示例数据和少量训练步数用于验证流程，生成质量需要充分训练才能评估。

## 安装

需要 Python 3.13 或更高版本，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在项目根目录执行：

```sh
uv sync --frozen
uv run pytest -q
```

锁文件固定依赖版本；运行依赖仅为 PyTorch 和 Transformers，开发组额外安装 pytest 和 Ruff。CPU 可以运行全部测试和小模型演示。训练默认自动选择 CUDA、MPS 或 CPU，使用 `--device cpu` 可显式选择 CPU。GPU 训练请按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)配置适合本机的 CUDA 版本。

## 快速跑通：预训练 → SFT → 推理

以下命令适用于 PowerShell、bash，均在仓库根目录运行。离线演示分词器使用 259 个 byte / special token，创建一次即可；再次创建时会拒绝覆盖已有文件。

```sh
# 1. 创建演示分词器；无需下载 tokenizer 或语料
uv run python scripts/create_demo_assets.py

# 2. 训练一个约 3.5 万参数的小模型，最多执行 4 次参数更新
uv run python trainer/train_pretrain.py --data-path examples/pretrain.jsonl --tokenizer-path assets/demo --run-name pretrain_demo --hidden-size 32 --intermediate-size 64 --num-hidden-layers 2 --num-attention-heads 4 --num-key-value-heads 2 --max-position-embeddings 512 --seq-length 64 --batch-size 2 --accumulation-steps 2 --epochs 2 --max-steps 4 --warmup-steps 0 --log-interval 1 --val-size 1 --device cpu

# 3. 使用预训练检查点微调；自动读取模型结构和分词器
uv run python trainer/train_sft.py --data-path examples/sft.jsonl --checkpoint output/pretrain_demo/4 --run-name sft_demo_small --seq-length 512 --batch-size 2 --accumulation-steps 2 --epochs 3 --max-steps 4 --warmup-steps 0 --log-interval 1 --val-size 2 --device cpu

# 4. 单轮推理；temperature=0 表示贪心生成
uv run python main.py --checkpoint output/sft_demo_small/4 --prompt "你好" --max-new-tokens 32 --temperature 0 --device cpu

# 5. 省略 prompt，进入交互对话；输入 exit 退出
uv run python main.py --checkpoint output/sft_demo_small/4 --max-new-tokens 32 --device cpu
```

训练结束会打印 JSON 摘要和检查点路径，指标写入 `loss.jsonl`，至少两个训练日志点时生成 `loss_curve.svg`。不要以演示模型的回复判断项目最终能力；byte tokenizer 的少量训练可能生成不可读文本。

`uv sync` 也安装以下等价入口：`uv run jxmind`、`uv run jxmind-pretrain`、`uv run jxmind-sft`、`uv run jxmind-download`、`uv run jxmind-demo`。完整参数可通过每个命令的 `--help` 查看。

## 使用实际语料

下载 MiniMind 分词器，默认保存到 `assets/`：

```sh
uv run python scripts/download_assets.py
# 可选：下载大型 SFT CSV 语料，需预留数 GB 磁盘空间
uv run python scripts/download_assets.py --with-data
```

网络不通时，可手动将兼容的 `tokenizer.json`、`tokenizer_config.json` 放入本地目录，再指定 `--tokenizer-path`。本地加载不会访问网络。分词器必须有 EOS token；SFT 和聊天还需要 `chat_template`。新模型默认从分词器确定词表大小。

数据支持 UTF-8 JSONL / CSV。预训练读取 `text`、`content` 或 `q/a`；每条文本后插入 EOS，拼成定长 token 块，末尾不足一个块的 token 丢弃。SFT 支持 `q/a`、`question/response`、`prompt/chosen`、`instruction/input/output` 和 `messages`，仅对最后一条助手回复计算 loss。字段示例见 [数据说明](docs/data.md)。

```sh
uv run python trainer/train_pretrain.py --data-path data/pretrain.jsonl --tokenizer-path assets --run-name pretrain_full --seq-length 512 --batch-size 8 --accumulation-steps 2 --epochs 1
uv run python trainer/train_sft.py --data-path data/sft_data_single.csv --checkpoint output/pretrain_full/500 --run-name sft_full --seq-length 512 --batch-size 8 --accumulation-steps 2 --epochs 1
```

第二条命令中的 `500` 是示例，替换为实际保存的检查点步数。数据编码后保存在内存中，大语料需要预留 RAM；可用 `--max-samples 10000` 限制读取条数。`--val-size` 是验证样本数上限，始终至少保留一个训练样本，`--val-size 0` 关闭验证。预训练按 token 块分割验证集，正式评估建议事先按文档划分独立语料，避免同一文档跨集合。

默认模型配置：hidden size 512、8 层、8 个 Q 头、2 个 KV 头、FFN 1408、最大位置表 32768。位置表长度是实现容量，实际长文本能力取决于训练数据。`--use-moe` 启用路由专家和共享专家；默认 4 个路由专家、1 个共享专家、每个 token 选择 2 个路由专家。

## 继续训练与检查点

`--checkpoint` 只加载权重，用于开启新的训练阶段；从目录加载时使用目录中的模型配置。新训练应指定新的 `--run-name`，避免混合旧日志。原 run 已存在时程序会报错，要求选择新名称或使用 `--resume`。

`--resume` 从同一个 run 的最新完整检查点恢复权重、优化器、随机状态、epoch、batch 和更新步数。例如，将上面的预训练扩展到 8 步：

```sh
uv run python trainer/train_pretrain.py --data-path examples/pretrain.jsonl --run-name pretrain_demo --resume --seq-length 64 --batch-size 2 --accumulation-steps 2 --epochs 3 --max-steps 8 --warmup-steps 0 --log-interval 1 --val-size 1 --device cpu
```

保持相同的数据、切分、tokenizer、batch size、累积步数和 seed。程序会校验数据和词表摘要。`--epochs` 和 `--max-steps` 是累计训练目标；`max-steps=0` 表示不设额外步数上限。修改目标长度会改变余下训练的余弦学习率曲线。

检查点结构和旧版本兼容说明见 [训练与检查点](docs/training.md)。旧版本产生的 `output/sft_demo` 等文件保留在本机，但旧训练的掩码存在未来信息泄漏；建议用修复后的代码开启新 run 重新训练。

## 推理与上下文

新检查点内包含分词器，可以复制整个检查点目录到其他机器推理。模型严格校验参数名称和形状。上游 MiniMind 权重的结构及参数名称会随版本变化，不能直接视为本项目检查点。

`--max-context` 限制输入历史长度，程序同时为 `--max-new-tokens` 预留位置空间。历史过长时删除最早的完整问答轮次，保留 system 提示和当前问题；当前问题仍过长时给出错误。`--system` 设置系统提示，`--top-k` / `--top-p` / `--temperature` 控制生成。`--from-scratch` 使用随机参数，仅适合检查接口。

## 项目结构

```text
main.py                       单轮与交互推理
model/model.py                模型、位置编码、注意力、MoE、生成
dataset/lm_dataset.py         数据读取、标签、padding、训练/验证切分
trainer/train_pretrain.py      预训练入口
trainer/train_sft.py           SFT 入口
trainer/trainer_utils.py       优化器、训练循环、检查点、日志、曲线
scripts/download_assets.py    外部资源下载
scripts/create_demo_assets.py 离线演示分词器
examples/                     小型示例语料
tests/                        数值、数据、训练及回归测试
docs/                         数据与训练说明
.github/workflows/ci.yml       Linux / Windows 测试与构建
```

`assets/`、`data/`、`output/`、虚拟环境和工具缓存均被 Git 忽略；示例语料、源码和文档正常纳入版本管理。已移除无关的 `pra.py` 练习文件和三张描述旧实现边界的图片。

## 开发与验证

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

测试不依赖外部语料或预下载的 tokenizer。覆盖 RMSNorm / RoPE / YaRN、SFT 标签与 padding、Dense / MoE 前向反向、因果性、KV 缓存、批量 EOS、上下文边界、梯度累积、短批次以及断点恢复。CI 配置在 Linux 和 Windows 上执行同样检查。

本地验证使用 Python 3.14.6、PyTorch 2.9.0 CPU、Transformers 4.57.1；CUDA / MPS 尚未在本次修复中实机验证。当前训练采用 FP32 和单设备；暂不包含 DDP、混合精度、RLHF 或服务部署。

## 致谢与许可

架构和位置编码参考 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)，配置字段和 YaRN 超参沿用其设计。本仓库是学习与复现项目，采用 [Apache-2.0](LICENSE)。外部 tokenizer 和数据由下载脚本按需取得，不随仓库分发；使用外部语料前请核对来源和许可。
