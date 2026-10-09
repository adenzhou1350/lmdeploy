# Copyright (c) OpenMMLab. All rights reserved.
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from lmdeploy.pytorch.config import DistConfig, TPMode
from lmdeploy.pytorch.distributed import DistContext, _build_mlp_tp_group, _build_tp_group_impl


@pytest.mark.parametrize('dp,mlp_tp,moe_tp', [(4, 2, 4), (4, 4, 8), (2, 2, 2)])
def test_mlp_gather_group_membership(monkeypatch, dp, mlp_tp, moe_tp):
    config = DistConfig(dp=dp, mlp_tp=mlp_tp, moe_tp=moe_tp)
    assert config.mlp_tp_mode == TPMode.DP_TP

    for rank in range(config.world_size):
        def new_group(ranks, **kwargs):
            members = tuple(ranks)
            return SimpleNamespace(ranks=members) if rank in members else dist.GroupMember.NON_GROUP_MEMBER

        monkeypatch.setattr(dist, 'new_group', new_group)
        context = DistContext(rank=rank, dist_config=config)
        _build_mlp_tp_group(context, timedelta(seconds=10))

        group = context.mlp_tp_group.gpu_gather_group
        start = rank // mlp_tp * mlp_tp
        expected = tuple(range(start + rank % config.attn_tp, start + mlp_tp, config.attn_tp))
        assert group != dist.GroupMember.NON_GROUP_MEMBER, f'rank {rank} selected another TP block'
        assert group.ranks == expected


@pytest.mark.parametrize('mode,attn_tp', [(TPMode.DEFAULT, 1), (TPMode.DP_TP, 2)])
def test_tp_gather_group_controls(monkeypatch, mode, attn_tp):
    for rank in range(4):
        def new_group(ranks, **kwargs):
            members = tuple(ranks)
            return SimpleNamespace(ranks=members) if rank in members else dist.GroupMember.NON_GROUP_MEMBER

        monkeypatch.setattr(dist, 'new_group', new_group)
        group = _build_tp_group_impl(2, rank, 4, timedelta(seconds=10), attn_tp=attn_tp, tp_mode=mode)
        if mode == TPMode.DEFAULT:
            assert group.gpu_gather_group is None
        else:
            assert group.gpu_gather_group is group.gpu_group
