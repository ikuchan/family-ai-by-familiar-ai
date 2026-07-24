"""話者未識別時のデフォルト companion_name は中立ラベル「推定話者」であること。

FAMILY.md 先頭メンバー（例：パパ）を既定話者に derive していた不具合（「初手からパパ」）の
是正。COMPANION_NAME env があればそれを尊重、無ければ gui_estimated_speaker（推定話者）。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent._i18n import _t
from familiar_agent.config import AgentConfig


def test_companion_name_defaults_to_estimated_speaker():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("COMPANION_NAME", None)
        assert AgentConfig().companion_name == _t("gui_estimated_speaker")


def test_companion_name_respects_env():
    with patch.dict(os.environ, {"COMPANION_NAME": "ママ"}, clear=False):
        assert AgentConfig().companion_name == "ママ"
