import pytest
import argparse
import unittest

from enum import Enum

from um2nc.cli import parse_args, EnumAction
from um2nc.stashmasters import STASHmaster


def test_parse_args_convert():
    """
    Check that arguments are parsed correctly when the 'convert' command is
    either included or omitted.
    """
    with unittest.mock.patch("sys.argv", ["um2nc", "input_file", "output_file"]):
        with pytest.warns(match="No command recognised among"):
            no_convert_args_out = parse_args()

    with unittest.mock.patch("sys.argv", ["um2nc", "convert", "input_file", "output_file"]):
        convert_args_out = parse_args()

    assert no_convert_args_out.command == convert_args_out.command == "convert"
    assert no_convert_args_out.infile == convert_args_out.infile == "input_file"
    assert no_convert_args_out.outfile == convert_args_out.outfile == "output_file"

    assert no_convert_args_out == convert_args_out


@pytest.mark.parametrize(
    "cli_args,expected_defaults",
    [
        (
            ["um2nc", "convert", "infile", "outfile"],
            set([
                ("simple", False),
                ("strict", False),
                ("verbose", False),
                ("model", STASHmaster.DEFAULT.value)
            ])
        ),
        (
            ["um2nc", "driver", "esm1p5", "output_dir"],
            set([
                ("simple", True),
                ("strict", True),
                ("verbose", True),
                ("model", STASHmaster.ACCESS_ESM1p5)
            ])
        )
    ]
)
def test_command_defaults(cli_args, expected_defaults):
    """
    Test that different default values are correctly set with the 'convert'
    and 'driver esm1p5' commands.
    """
    with unittest.mock.patch("sys.argv", cli_args):
        args = parse_args()
    args_set = set(vars(args).items())
    assert expected_defaults.issubset(args_set)


# Tests of the EnumAction
@pytest.fixture
def test_enum():
    class Test(Enum):
        ON = "on"
        OFF = "off"
    return Test

@pytest.fixture
def enum_parser(test_enum):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enum", 
        type=test_enum, 
        action=EnumAction
    )
    return parser


def test_enum_action_valid_input(enum_parser, test_enum):
    """
    Test that the EnumAction returns the correct value for valid input.
    """
    args = enum_parser.parse_args(["--enum", "on"])
    assert args.enum is test_enum.ON


def test_enum_action_invalid_input(enum_parser):
    """
    Test that the EnumAction raises an error for a not valid input.
    """
    with pytest.raises(SystemExit):
        enum_parser.parse_args(["--enum", "fake_enum"])


def test_enum_action_choices(enum_parser, test_enum):
    """
    Test that the EnumAction sets the correct choices.
    """
    enum_action = [act for act in enum_parser._actions if act.dest == "enum"][0]
    assert enum_action.choices == tuple(c.value for c in test_enum)


def test_enum_action_choices_set(test_enum):
    """
    Test that the EnumAction raises a ValueError if 'choices' keyword is supplied.
    """
    parser = argparse.ArgumentParser()
    with pytest.raises(ValueError):
        parser.add_argument(
            "--enum",
            type=test_enum,
            choices=['fake','choices',1,None],
            action=EnumAction,
        )


def test_enum_action_no_enum_type():
    """
    Test that the EnumAction raises an error if type is not Enum.
    """
    parser = argparse.ArgumentParser()
    with pytest.raises(TypeError):
        parser.add_argument(
            "--enum", 
            type=str, 
            action=EnumAction
        )
    with pytest.raises(TypeError):
        parser.add_argument(
            "--enum2",
            action=EnumAction
        )
