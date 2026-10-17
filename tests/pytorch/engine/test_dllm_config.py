# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


@pytest.mark.parametrize(('block_length', 'prefill_budget'), [
    (-1, 8192),
    (0, 8192),
    (1, 0),
    (8193, 8192),
    (16384, 8192),
])
def test_invalid_dllm_block_length_raises(block_length, prefill_budget):
    config = PytorchEngineConfig(max_batch_size=64,
                                dllm_block_length=block_length,
                                max_prefill_token_num=prefill_budget)

    with pytest.raises(ValueError, match='dllm_block_length.*max_prefill_token_num'):
        ConfigBuilder.update_engine_config(config)


@pytest.mark.parametrize(('block_length', 'prefill_budget', 'expected_batches'), [
    (None, 8192, 64),
    (4, 8192, 64),
    (1024, 8192, 8),
    (8192, 8192, 1),
    (1024, 1024, 1),
    (4, 7, 1),
])
def test_dllm_batch_clamp_preserves_valid_configs(block_length, prefill_budget, expected_batches):
    config = PytorchEngineConfig(max_batch_size=64,
                                dllm_block_length=block_length,
                                max_prefill_token_num=prefill_budget)

    normalized = ConfigBuilder.update_engine_config(config)

    assert normalized.max_batch_size == expected_batches
    assert ConfigBuilder.build_scheduler_config(normalized).max_batches == expected_batches
    assert config.max_batch_size == 64
