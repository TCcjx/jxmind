"""Create an offline byte tokenizer for small training fixtures."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def create_tokenizer(path):
    path = Path(path)
    if any(path.glob("tokenizer*.json")):
        raise FileExistsError(f"refusing to overwrite tokenizer in {path}")
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(special + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=special[0]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=special[0],
        bos_token=special[1],
        eos_token=special[2],
        chat_template=CHAT_TEMPLATE,
    )
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets/demo")
    args = parser.parse_args()
    tokenizer = create_tokenizer(args.output_dir)
    print(json.dumps({"tokenizer": args.output_dir, "vocab_size": len(tokenizer)}))


if __name__ == "__main__":
    main()
