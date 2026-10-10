# 训练与检查点

## 更新与学习率

训练使用 AdamW。二维及更高维权重参与 weight decay，norm / bias 不参与。梯度累积按实际被监督的 next-token 数加权，与同样数据组成整批更新时的目标一致。epoch 尾部不满一个累积窗口的梯度仍会更新，不会带到下个 epoch。

学习率为线性 warmup + 余弦衰减；step 从 0 开始计算，再更新参数。`--warmup-steps` 默认 50，短演示可设 0。`--max-steps` 限制累计优化器更新次数，仍受 epochs 覆盖的数据量限制。`--num-threads` 默认 4，可调整 CPU 运算线程数。

MoE loss 包含各层路由负载均衡辅助项。`seq_aux=True` 按每条序列计算均衡损失；训练日志和验证 loss 都包含该辅助项。MoE 的路由统计包含 padding token，小模型实验请留意不同 padding 比例的影响。

## 输出

```text
output/<run-name>/
  loss.jsonl                 训练/验证指标
  loss_curve.svg             至少 2 个训练日志点时生成
  samples.txt                开启 --sample-interval 和 --sample-prompt 时生成
  <step>/
    model.pth                state_dict 权重
    opt.pth                  AdamW 状态
    rng.pth                  Torch / CUDA 随机状态
    config.json              模型配置
    meta.json                更新步数、下次 epoch/batch、数据摘要、训练配置
    tokenizer/               tokenizer.json 和 tokenizer_config.json 等
```

`--log-interval` 默认 10，第一步和最后一步总会记录。`--eval-interval` 默认 200，`--save-interval` 默认 500；设为 0 可关闭对应周期操作。训练结束仍会验证并保存检查点。`--sample-interval` 默认 0，可配合重复的 `--sample-prompt` 开启采样。

## 加载与恢复

- `--checkpoint <目录>`：加载目录中的模型配置、权重和分词器，以新优化器启动训练阶段。架构参数以检查点配置为准。
- `--checkpoint <model.pth>`：只加载权重；需要显式提供匹配的模型结构及分词器。
- `--resume --run-name <名称>`：恢复该 run 中最新检查点的完整状态。保持数据、切分、batch size、累积步数、seed 和 worker 数不变。程序校验编码数据与词表摘要，避免接续到不同语料。

更改 epochs / max-steps 可以延长训练；改变累计目标长度也会改变学习率调度。CPU、同样配置与 num_workers=0 下的中断恢复由测试验证与连续训练一致。更换设备或底层 kernel 可能产生不同的浮点结果。

没有 cursor / rng 的旧检查点可以用于权重初始化和推理，不能完整恢复训练。参数名称或形状不匹配时加载会报错。不要将上游 MiniMind 的任意权重文件直接传入本项目。

## 模型说明

模型为 Pre-norm Decoder-only Transformer：Embedding → 多层 GQA + SwiGLU / MoE → RMSNorm → LM head。RMSNorm 和 RoPE 计算中间量使用 float32，再转换回输入 dtype。SDPA 路径和手写注意力均组合 causal 与 padding 掩码；可用 `--no-flash-attention` 切到手写实现。SDPA 的实际 kernel 由 PyTorch 按设备和输入选择，参见 [官方说明](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)。

YaRN 可通过 `MokioMindConfig(inference_rope_scaling=True)` 或自定义 `rope_scaling` 配置启用，JSON 保存和加载会保留缩放参数。频率实现的数值校验及旧的实测曲线如下：

![RoPE / YaRN 频率曲线](images/rope_yarn.png)

曲线使用 head_dim=64、rope_base=1e6、original_max=2048、factor=16。高频部分保留，低频部分压缩到 1/16。这张曲线说明位置编码数值，不代表训练后模型的长上下文质量。
