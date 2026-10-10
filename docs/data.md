# 数据格式

输入为 UTF-8（允许 BOM）的 JSONL 或 CSV。JSONL 每行必须是一个对象。CSV 第一行为字段名。训练前完成分词并将 token 保存到内存；预训练逐条分词，不会先拼接一份完整字符串。

## 预训练

```json
{"text": "语言模型预测下一个词元。"}
{"content": "注意力机制计算信息之间的相关性。"}
{"q": "水的化学式？", "a": "H2O"}
```

字段优先级为 `text > content > q/a`。各条文本用 EOS 分隔，切成 `--seq-length` 长度的块，labels 与 input_ids 对齐，模型内部右移一位计算 next-token loss。最后不足一个块的 token 丢弃；语料不足一个块时明确报错。

## 指令微调

```json
{"q":"你好", "a":"你好！", "history":[["你是谁？","我是一个助手。"]]}
{"question":"一年有几个月？", "response":"十二个月。"}
{"prompt":"解释梯度", "chosen":"梯度描述函数变化的方向与幅度。"}
{"instruction":"翻译成中文", "input":"Hello", "output":"你好"}
{"messages":[{"role":"system","content":"简短回答。"},{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}
```

`history` 可以是列表，也可以是 JSON 序列化后的字符串。支持 `q/a`、`question/response`、`prompt/chosen` 和 `instruction/input/output`；`input` 拼到 instruction 后。可选 `system` 字段作为系统消息。

`messages` 中支持 `system`、`user`、`assistant`，content 必须是字符串，最后一条必须是 assistant。使用 tokenizer 的聊天模板编码，只监督最后一条助手回复及模板结尾。用户问题、system、历史助手回复和 padding 的 labels 为 `-100`。

超出序列长度的样本从右侧截断；截断后没有可预测的回答 token、缺少问答或最后一条不是 assistant 的样本会被跳过。所有样本都不可用时会报错。训练入口打印保留样本数和跳过数量。

## 切分和 padding

`--seed` 控制索引打乱与训练 shuffle。验证集样本数为 `min(val_size, N - 1)`；N=1 或 val_size=0 时使用空验证集。训练尾部不足 batch size 的样本仍参与更新。

一个 batch 按最长样本右侧 padding，attention_mask 为 0 的位置被屏蔽，labels 为 `-100`。padding 掩码与因果掩码共同生效。只支持未填充 prompt 的生成接口；CLI 使用单条输入。

演示文件 [sft.jsonl](../examples/sft.jsonl) 和 [pretrain.jsonl](../examples/pretrain.jsonl) 只用于检查运行流程。
