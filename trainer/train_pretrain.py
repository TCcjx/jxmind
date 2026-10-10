"""预训练入口：以 next-token 目标在纯文本语料上训练 MokioMind。

用法：
    uv run python trainer/train_pretrain.py --data-path data/sft_data_single.csv --seq-length 256 --epochs 1
语料字段优先级 text > content > q/a，因此 MiniMind 的 SFT csv 也可直接用于预训练。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.lm_dataset import PretrainDataset, split_dataset
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
    parser = argparse.ArgumentParser(description="MokioMind 预训练")
    add_arguments(parser)
    args = parser.parse_args()

    tokenizer = tokenizer_from_args(args)
    config = config_from_args(args, tokenizer)
    cfg = train_config_from_args(args)
    dataset = PretrainDataset(
        args.data_path, tokenizer, sequence_length=args.seq_length, limit=args.max_samples or None
    )
    train_ds, val_ds = split_dataset(dataset, val_size=args.val_size, seed=args.seed)
    model = init_model(config, args.checkpoint)
    run_name = args.run_name or f"pretrain_{human(trainable_params(model))}"
    run_dir = RunDirectory(args.output_dir, run_name)

    print(f"[data] blocks={len(dataset)} train={len(train_ds)} val={len(val_ds)} seq={args.seq_length}")
    print(f"[model] {human(trainable_params(model))} params, moe={config.use_moe}")
    summary = run_training(model, tokenizer, train_ds, val_ds, cfg, run_dir)
    summary["run_dir"] = run_dir.root
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
