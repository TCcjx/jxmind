"""推理入口：加载 checkpoint 与训练好的模型进行对话或单轮生成。

用法：
    uv run python main.py --checkpoint output/sft_15.6M/120
    uv run python main.py --checkpoint output/sft_15.6M/120 --prompt "用一句话解释注意力机制"
    uv run python main.py --from-scratch        # 随机初始化，仅用于冒烟测试
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from dataset.lm_dataset import load_tokenizer
from trainer.trainer_utils import configure_stdio, init_model, pick_device, read_model_config


def build_parser():
    parser = argparse.ArgumentParser(description="MokioMind 推理")
    parser.add_argument("--checkpoint", type=str, default=None, help="含 model.pth + meta.json 的目录")
    parser.add_argument("--from-scratch", action="store_true", help="不载入权重，用随机初始化的模型")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="单轮生成；省略则进入交互模式")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-context", type=int, default=1024, help="超长时从最早一轮开始丢弃历史")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def fit_context(tokenizer, history, max_context):
    """交互模式会无限追加历史，这里按 token 数截断，避免超出位置编码表。"""
    first = 1 if history and history[0]["role"] == "system" else 0
    while len(history) > first + 1:
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_context:
            break
        del history[first : first + 2]
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_context:
        raise ValueError("当前问题和 system 提示过长，请缩短内容或增大 --max-context")
    return history


@torch.no_grad()
def respond(model, tokenizer, messages, device, args):
    budget = min(args.max_context, model.config.max_position_embeddings - args.max_new_tokens)
    messages = fit_context(tokenizer, messages, budget)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    generated = model.generate(
        encoded["input_ids"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def main():
    configure_stdio()
    args = build_parser().parse_args()
    if bool(args.checkpoint) == args.from_scratch:
        sys.exit("请指定 --checkpoint，或使用 --from-scratch 做冒烟测试")
    if args.max_new_tokens < 1 or args.max_context < 1 or args.num_threads < 1:
        sys.exit("--max-new-tokens、--max-context 和 --num-threads 必须为正数")
    torch.set_num_threads(args.num_threads)

    device = args.device or pick_device()
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path and args.checkpoint:
        bundled = os.path.join(args.checkpoint, "tokenizer")
        if os.path.isdir(bundled):
            tokenizer_path = bundled
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else load_tokenizer()

    if args.checkpoint:
        config = read_model_config(args.checkpoint)
        model = init_model(config, args.checkpoint, device)
        print(f"[model] {args.checkpoint} | {sum(p.numel() for p in model.parameters()):,} params | {device}")
    else:
        from model.model import MokioMindConfig

        model = init_model(
            MokioMindConfig(
                vocab_size=len(tokenizer),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ),
            None,
            device,
        )
        print(f"[model] 随机初始化（仅冒烟测试） | {device}")
    model.eval()
    if model.config.vocab_size != len(tokenizer):
        sys.exit("模型词表大小与 tokenizer 不匹配")

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    if args.prompt:
        history.append({"role": "user", "content": args.prompt})
        print(respond(model, tokenizer, history, device, args))
        return

    print("输入问题回车对话，exit 退出。")
    while True:
        try:
            question = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "bye"}:
            break
        if not question:
            continue
        history.append({"role": "user", "content": question})
        try:
            answer = respond(model, tokenizer, history, device, args)
        except ValueError as exc:
            history.pop()
            print(f"[error] {exc}")
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"BOT> {answer}\n")


if __name__ == "__main__":
    main()
