# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference imports must not require removed training logger exports."""
import importlib

import pytest


@pytest.mark.unit
def test_exp_manager_without_neptune_logger_export(monkeypatch):
    import lightning.pytorch.loggers as loggers

    monkeypatch.delattr(loggers, 'NeptuneLogger', raising=False)
    module = importlib.import_module('nemo.utils.exp_manager')
    module = importlib.reload(module)
    assert callable(module.exp_manager)
