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


def test_model_flag_parses_on_run_and_bench():
    assert build_parser().parse_args([]).model is None
    assert build_parser().parse_args(["run", "--model", "x/y"]).model == "x/y"
    assert build_parser().parse_args(["bench", "--model", "/tmp/m"]).model == "/tmp/m"


def test_bare_flags_route_to_run_subcommand():
    from localvoice.__main__ import parse_cli

    args = parse_cli(["--think"])
    assert args.command == "run" and args.think is True
    args = parse_cli(["--model", "x/y", "--deep"])
    assert args.command == "run" and args.model == "x/y" and args.deep is True
    assert parse_cli([]).command == "run"
    assert parse_cli(["bench", "--runs", "5"]).runs == 5
    assert parse_cli(["setup"]).command == "setup"


def test_serve_parses():
    args = build_parser().parse_args(["serve", "--allow-inject"])
    assert args.command == "serve" and args.allow_inject is True
    assert build_parser().parse_args(["serve"]).allow_inject is False
