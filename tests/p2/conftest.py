# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_threaded_cpu_oracles():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
