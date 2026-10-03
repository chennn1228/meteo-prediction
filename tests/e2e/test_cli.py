from __future__ import annotations

from nwp.cli import main


def test_single_cli_can_validate_a_resolved_profile_without_creating_a_run():
    assert main(["validate", "--profile", "nanjing_cpu_diagnostic"]) == 0
