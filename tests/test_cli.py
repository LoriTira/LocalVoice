from localvoice.__main__ import build_parser


def test_default_command_is_run():
    args = build_parser().parse_args([])
    assert args.command == "run" and args.deep is False and args.think is False


def test_run_flags():
    args = build_parser().parse_args(["run", "--deep", "--think", "--config", "x.toml"])
    assert args.deep and args.think and args.config == "x.toml"


def test_setup_and_bench_parse():
    assert build_parser().parse_args(["setup"]).command == "setup"
    args = build_parser().parse_args(["bench", "--runs", "5"])
    assert args.command == "bench" and args.runs == 5
