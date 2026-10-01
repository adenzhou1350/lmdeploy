# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def kernel_utils(monkeypatch):
    path = Path(__file__).parents[3] / 'lmdeploy/pytorch/kernels/cuda/utils.py'
    spec = importlib.util.spec_from_file_location('cuda_kernel_utils_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'is_cuda', lambda: True)
    return module


@pytest.mark.parametrize('majors', [(8, 9), (9, 8)])
def test_supports_pdl_follows_current_device(kernel_utils, monkeypatch, majors):
    current = [0]
    caps = {0: (majors[0], 0), 1: (majors[1], 0)}
    monkeypatch.setattr(kernel_utils.torch.cuda, 'current_device', lambda: current[0])
    monkeypatch.setattr(kernel_utils.torch.cuda, 'get_device_capability',
                        lambda device=None: caps[current[0] if device is None else device])
    monkeypatch.setattr(kernel_utils.torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(major=caps[device][0], minor=caps[device][1],
                                                       multi_processor_count=1))
    assert kernel_utils.supports_pdl() is (majors[0] >= 9)
    current[0] = 1
    assert kernel_utils.supports_pdl() is (majors[1] >= 9)
    current[0] = 0
    assert kernel_utils.supports_pdl() is (majors[0] >= 9)


@pytest.mark.parametrize('major, expected', [(8, False), (9, True), (10, True), (12, True)])
def test_supports_pdl_explicit_device(kernel_utils, monkeypatch, major, expected):
    queries = []

    def properties(device):
        queries.append(device)
        assert device == 2
        return SimpleNamespace(major=major, minor=0, multi_processor_count=1)

    monkeypatch.setattr(kernel_utils.torch.cuda, 'get_device_properties', properties)
    monkeypatch.setattr(kernel_utils.torch.cuda, 'current_device', lambda: pytest.fail('Unexpected CUDA query'))
    assert kernel_utils.supports_pdl(2) is expected
    assert kernel_utils.supports_pdl(2) is expected
    assert queries == [2]


def test_supports_pdl_non_cuda(kernel_utils, monkeypatch):
    monkeypatch.setattr(kernel_utils, 'is_cuda', lambda: False)
    monkeypatch.setattr(kernel_utils.torch.cuda, 'current_device', lambda: pytest.fail('Unexpected CUDA query'))
    monkeypatch.setattr(kernel_utils.torch.cuda, 'get_device_capability', lambda: pytest.fail('Unexpected CUDA query'))
    assert kernel_utils.supports_pdl() is False
