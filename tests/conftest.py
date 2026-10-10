import pytest
import torch

from scripts.create_demo_assets import create_tokenizer


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    return create_tokenizer(tmp_path_factory.mktemp("demo_tokenizer"))
