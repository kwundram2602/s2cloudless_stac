from pathlib import Path

from s2cloudless_stac.cli import parse_args, run_all


def test_run_all_continues_after_failure_and_reports_it():
    done = []

    def process(item_id: str) -> None:
        if item_id == "bad":
            raise LookupError("no L1C")
        done.append(item_id)

    assert run_all(["a", "bad", "b"], process) == 1
    assert done == ["a", "b"]


def test_run_all_success_exit_code():
    assert run_all(["a"], lambda _: None) == 0


def test_parse_args_defaults():
    args = parse_args(["S2C_MSIL2A_x"])
    assert args.ids == ["S2C_MSIL2A_x"]
    assert args.out_dir == Path("data/out")
    assert args.threshold == 0.4
    assert args.average_over == 4
    assert args.dilation_size == 2
    assert args.allow_baseline_mismatch is False
