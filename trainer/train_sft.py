"""SFT 入口：在指令对话数据上微调，只在 assistant 回复 span 上计算 loss。

用法：
    uv run python trainer/train_sft.py --data-path data/sft_data_single.csv --seq-length 256
可从预训练权重继续：追加 --checkpoint output/pretrain_xxx/<step>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.lm_dataset import SupDataset, split_dataset
from trainer.trainer_utils import (
    RunDirectory,
    add_arguments,
    config_from_args,
    configure_stdio,
    human,
    init_model,
    run_training,
    tokenizer_from_args,
    train_config_from_args,
    trainable_params,
)


def main():
    configure_stdio()
    parser = argparse.ArgumentParser(description="MokioMind 指令微调")
    add_arguments(parser)
    args = parser.parse_args()

    tokenizer = tokenizer_from_args(args)
    config = config_from_args(args, tokenizer)
    cfg = train_config_from_args(args)
    dataset = SupDataset(
        args.data_path, tokenizer, sequence_length=args.seq_length, limit=args.max_samples or None
    )
    train_ds, val_ds = split_dataset(dataset, val_size=args.val_size, seed=args.seed)
    model = init_model(config, args.checkpoint)
    run_name = args.run_name or f"sft_{human(trainable_params(model))}"
    run_dir = RunDirectory(args.output_dir, run_name)

    print(f"[data] samples={len(dataset)} skipped={dataset.skipped} train={len(train_ds)} val={len(val_ds)}")
    print(f"[model] {human(trainable_params(model))} params, moe={config.use_moe}")
    summary = run_training(model, tokenizer, train_ds, val_ds, cfg, run_dir)
    summary["run_dir"] = run_dir.root
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
