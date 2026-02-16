import argparse
import copy
import logging
import sys
import warnings

from enum import Enum

import um2nc
from um2nc.stashmasters import STASHmaster
from um2nc.um2netcdf import process, StrictWarning
from um2nc.drivers.esm1p5 import convert_esm1p5_output_dir


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums.
    It automatically produces choices based on the Enum values.
    """

    def __init__(self, **kwargs):
        # If 'choices' were declared explicitely, raise an error
        if "choices" in kwargs:
            raise ValueError(
                f"Cannot use 'choices' keyword together with {self.__class__.__name__}. "
                f"Choices are automatically generated from the Enum values."
            )
        # Pop the 'type' keyword
        enum_type = kwargs.pop("type", None)
        # Ensure an Enum subclass is provided
        if not enum_type or not issubclass(enum_type, Enum):
            raise TypeError(
                f"The 'type' keyword must be assigned to Enum (or any Enum subclass) when using {self.__class__.__name__}."
            )
        # Generate choices from the Enum values
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))
        # Call the argparse.Action constructor with the remaining keyword arguments
        super().__init__(**kwargs)
        # Store Enum subclass for use in the __call__ method
        self._enum = enum_type

    def __call__(self, parser, namespace, value, option_string=None):
        # Convert value to the associated Enum member
        member = self._enum(value)
        setattr(namespace, self.dest, member)


# Define arguments

# Arguments for the convert command which are also shared with the drivers
convert_args = argparse.ArgumentParser(add_help=False)

convert_args.add_argument(
    "-k",
    dest="nckind",
    required=False,
    type=int,
    default=3,
    choices=[1, 2, 3, 4],
    help=(
        "NetCDF output format. Choose among '1' (classic), '2' (64-bit offset),"
        "'3' (netCDF-4), '4' (netCDF-4 classic)."
    )
)
convert_args.add_argument(
    "-c", "--compression",
    dest="compression",
    required=False,
    type=int,
    default=4,
    help="Compression level. '0' (none) to '9' (max).",
)
convert_args.add_argument(
    "--64",
    dest="use64bit",
    action="store_true",
    default=False,
    help="Write 64 bit output when input is 64 bit"
)

restrict_group = convert_args.add_mutually_exclusive_group()
restrict_group.add_argument(
    "--include",
    dest="include_list",
    type=int,
    nargs="+",
    help="List of stash codes to include"
)
restrict_group.add_argument(
    "--exclude",
    dest="exclude_list",
    type=int,
    nargs="+",
    help="List of stash codes to exclude"
)

convert_args.add_argument(
    "--nomask",
    dest="nomask",
    action="store_true",
    default=False,
    help="Don't mask variables on pressure level grids.",
)
convert_args.add_argument(
    "--nohist",
    dest="nohist",
    action="store_true",
    default=False,
    help="Don't add/update the global 'history' attribute in the output netCDF."
)
convert_args.add_argument(
    "--hcrit",
    dest="hcrit",
    type=float,
    default=0.5,
    help="Critical value of the Heaviside variable for pressure level masking."
)
convert_args.add_argument(
    "--simple",
    dest="simple",
    action="store_true",
    help="Write output using simple variable names of format 'fld_s<section number>i<item number>'."
)
convert_args.add_argument(
    "--strict",
    dest="strict",
    action="store_true",
    default=False,
    help="Promote the StrictWarning class of warnings to errors.")

verbosity_group = convert_args.add_mutually_exclusive_group()
verbosity_group.add_argument(
    "-v", "--verbose",
    dest="verbose",
    action="store_true",
    help="Display verbose output."
)
verbosity_group.add_argument(
    "-q", "--quiet",
    dest="quiet",
    action="store_true",
    default=False,
    help="Suppress warnings arising from um2nc."
)

convert_args.add_argument(
    "--model",
    dest="model",
    type=STASHmaster,
    action=EnumAction,
    help=(
        "Link STASH codes to variable names and metadata by using a preset STASHmaster associated with a specific model. "
        f"Options: {[v.value for v in STASHmaster]}."
    ),
)

# Arguments shared between model drivers
driver_args = argparse.ArgumentParser(add_help=False)

driver_args.add_argument(
    "current_output_dir",
    help="Output directory to be converted",
    type=str
)
driver_args.add_argument(
    "--delete-ff", "-d",
    action="store_true",
    default=False,
    help="Delete fields files upon successful conversion"
)


# Set up parsers
parser = argparse.ArgumentParser(prog="um2nc", exit_on_error=False)
parser.add_argument(
    "--version","-V",
    action="version",
    version=um2nc.__version__,
)

subparsers = parser.add_subparsers(dest="command", required=True)

# convert subcommand
convert = subparsers.add_parser("convert", parents=[copy.deepcopy(convert_args)],  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
convert.add_argument("infile", help="Input file")
convert.add_argument("outfile", help="Output file")
# Set defaults which are specific for the convert command
convert.set_defaults(simple=False, strict=False, verbose=False, model=STASHmaster.DEFAULT.value)

# driver subcommand
driver = subparsers.add_parser("driver")

# esm1p5
model_subparsers = driver.add_subparsers(dest="command", required=True)
esm1p5 = model_subparsers.add_parser("esm1p5", parents=[copy.deepcopy(convert_args), copy.deepcopy(driver_args)],  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
esm1p5.set_defaults(simple=True, strict=True, verbose=True, model=STASHmaster.ACCESS_ESM1p5.value)


def setup_logging(verbose: bool, quiet: bool, strict: bool):
    """Setup logging levels"""
    level = logging.WARNING

    if verbose:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR  # Suppress WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Route warnings → logging
    logging.captureWarnings(True)

    # Apply strict option
    if strict:
        warnings.filterwarnings("error", category=StrictWarning)


def parse_args(argv=None):
    in_args = argv if argv is not None else sys.argv[1:]
    # Allow for the 'convert' command to be ommited by adding it if missing.
    try:
        args = parser.parse_args(in_args)
    except argparse.ArgumentError as e:
        if str(e).startswith('argument command: invalid choice:'):
            args = parser.parse_args(['convert'] + in_args)
        else:
            raise e

    return args


def main():
    args = parse_args()
    # Setup logging
    setup_logging(args.verbose, args.quiet, args.strict)

    # Run selected command
    if args.command == "convert":
        process(args.infile, args.outfile, args)
    elif args.command == "esm1p5":
        convert_esm1p5_output_dir(args.current_output_dir, args)
    else:
        raise RuntimeError(
            f"Unrecognised command {args.command}."
        )
