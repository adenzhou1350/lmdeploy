# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.messages import HistoryMultiModals
from lmdeploy.pytorch.multimodal.data_type import MultiModalData


@pytest.mark.parametrize(
    ('counts', 'expected'),
    [
        (None, True),
        ({}, True),
        ({'image': 0}, True),
        ({'image': 0, 'audio': 0}, True),
        ({'image': 1}, False),
        ({'image': 0, 'audio': 1}, False),
    ],
    ids=['none', 'empty-map', 'empty-image', 'empty-modalities', 'image', 'audio'],
)
def test_history_multimodals_empty_checks_contents(counts, expected):
    multimodals = None if counts is None else {
        kind: [MultiModalData(data=torch.zeros(1, device='cpu'), start=0, end=1) for _ in range(count)]
        for kind, count in counts.items()
    }
    assert HistoryMultiModals(multimodals).empty() is expected


def test_history_multimodals_empty_after_adding_empty_inputs():
    history = HistoryMultiModals()
    history.add_inputs({'image': [], 'audio': []})
    assert history.empty()
