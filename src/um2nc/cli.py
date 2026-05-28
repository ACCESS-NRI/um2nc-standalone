import argparse
import copy
import logging
import sys
import warnings

from enum import Enum
from pathlib import Path

import um2nc
from um2nc.common import DelayedCubePath
from um2nc.stashmasters import STASHmaster
from um2nc.um2netcdf import process, StrictWarning
from um2nc.drivers.esm1p5 import Esm1p5Driver
from um2nc.drivers.esm1p6 import Esm1p6Driver


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
common_args = argparse.ArgumentParser(add_help=False)

common_args.add_argument(
    "-f", "--format", "--nc-format", "--k",
    dest="ncformat",
    required=False,
    type=int,
    default=3,
    choices=[1, 2, 3, 4],
    help=(
        "NetCDF output format. Choose among '1' (classic), '2' (64-bit offset),"
        "'3' (netCDF-4), '4' (netCDF-4 classic)."
    )
)
common_args.add_argument(
    "-c", "--compression",
    dest="compression",
    required=False,
    type=int,
    default=4,
    help="NetCDF compression level. '0' (none) to '9' (max).",
)
common_args.add_argument(
    "--64",
    dest="use64bit",
    action="store_true",
    default=False,
    help="Write 64 bit output when input is 64 bit. When absent, output will be 32 bit."
)

restrict_group = common_args.add_mutually_exclusive_group()
restrict_group.add_argument(
    "--include",
    dest="include_list",
    type=int,
    nargs="+",
    help=("List of variables to include in the output file. Variables are specified by their STASH codes "
          "in the format '1000 * section number + item number'. "
          "No other variables will be written to the output file. "
          "Cannot be used with '--exclude'.")
)
restrict_group.add_argument(
    "--exclude",
    dest="exclude_list",
    type=int,
    nargs="+",
    help=("List of variables to exclude from the output file. Variables are specified by their STASH codes "
          "in the format '1000 * section number + item number'. " 
          "All other variables present in the input file will be written to the output file. "
          "Cannot be used with '--include'.")
)

common_args.add_argument(
    "--nomask",
    dest="nomask",
    action="store_true",
    default=False,
    help=("Data on a pressure level grid may fall below ground level during a simulation. "
          "By default, um2nc applies a heaviside mask to these points to ensure valid data is written to the "
          "output, and drops these variables if the mask variable is missing from the input. "
          "When selected, '--nomask' disables the masking and writes pressure level variables to the "
          "output without corrections.")
)
common_args.add_argument(
    "--hcrit",
    dest="hcrit",
    type=float,
    default=0.5,
    help=("Minimum fraction of the time spent above ground-level for a pressure grid data point "
          "to be considered valid. Data points in pressure grid variables will be masked if they "
          "were above ground-level for less than the critical fraction HCRIT of the time. "
          "This option has no effect when used together with the '--nomask' option."
    )
)
common_args.add_argument(
    "--model",
    dest="model",
    type=STASHmaster,
    action=EnumAction,
    help=(
        "Link STASH codes to variable names and metadata by using a preset STASHmaster associated with a specific model. "
        f"Options: {[v.value for v in STASHmaster]}. If omitted, the '{STASHmaster.CMIP6.value}' STASHmaster will be used."
    ),
)
common_args.add_argument(
    "--nohist",
    dest="nohist",
    action="store_true",
    default=False,
    help=("Don't add a global history attribute to the output file. By default, the conversion time, um2nc version, "
          "and script location will be added.")
)
common_args.add_argument(
    "--simple",
    dest="simple",
    action="store_true",
    help="Write output using simple variable names of format 'fld_s<section number>i<item number>'."
)
common_args.add_argument(
    "--one-nc-per-stash-variable",
    action="store_true",
    help="Create a separate netCDF file for each STASH code. The name for each "
        "STASH variable followed by an underscore will be used to prefix each "
        "file name."
)
common_args.add_argument(
    "--strict",
    dest="strict",
    action="store_true",
    default=False,
    help="Promote the StrictWarning class of warnings to errors."
)
verbosity_group = common_args.add_mutually_exclusive_group()
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


# Arguments for the convert command
convert_args = argparse.ArgumentParser(add_help=False)
convert_args.add_argument("infile", help="Input UM data file.")
convert_args.add_argument("outfile", help="Output netCDF file.")

# Arguments shared between model drivers
driver_args = argparse.ArgumentParser(add_help=False)

driver_args.add_argument(
    "model_directory",
    type=str,
    help="Path to a simulation's output directory containing UM files for conversion.",
)
driver_args.add_argument(
    "--delete-ff", "-d",
    action="store_true",
    default=False,
    help="Delete input files upon successful conversion."
)


# Set up parsers
parser = argparse.ArgumentParser(
    prog="um2nc",
    exit_on_error=False,
    description=(
        "Utilities for converting UM data files to netCDF. Use "
        "'um2nc {subcommand} --help' for usage information on each subcommand."
    )
)
parser.add_argument(
    "--version","-V",
    action="version",
    version=um2nc.__version__,
)

subparsers = parser.add_subparsers(dest="command", required=True)

# convert subcommand
convert = subparsers.add_parser(
    "convert",
    parents=[copy.deepcopy(common_args), convert_args],
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Convert a single input UM data file to netCDF.",
    help="Convert a single input UM data file to netCDF."
)
# Set defaults which are specific for the convert command
convert.set_defaults(simple=False, strict=False, verbose=False, model=STASHmaster.DEFAULT.value)

# driver subcommand
driver = subparsers.add_parser(
    "driver",
    help=(
        "Run a model driver for netCDF conversion during ACCESS model simulations."
    ),
    description=(
        "Run a model driver for automatic UM file to netCDF conversion during "
        "ACCESS model simulations. Use 'um2nc driver {model_driver} --help' "
        "for usage information for a specific model driver."
    )
)

# esm1p5
driver_subparsers = driver.add_subparsers(dest="model_driver", required=True)
esm1p5 = driver_subparsers.add_parser(
    "esm1p5",
    description=(
        "Model driver for automatic UM file to netCDF conversion "
        "during ACCESS-ESM1.5 simulations."
    ),
    help="Model driver for ACCESS-ESM1.5 netCDF conversion.",
    parents=[copy.deepcopy(common_args), copy.deepcopy(driver_args)],
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
esm1p5.set_defaults(simple=True, strict=True, verbose=True, model=STASHmaster.ACCESS_ESM1p5.value)

# esm1p6
esm1p6 = driver_subparsers.add_parser(
    "esm1p6",
    description=(
        "Model driver for automatic UM file to netCDF conversion "
        "during ACCESS-ESM1.6 simulations."
    ),
    help="Model driver for ACCESS-ESM1.6 netCDF conversion.",
    parents=[copy.deepcopy(common_args), copy.deepcopy(driver_args)],
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
esm1p6.set_defaults(simple=True, strict=True, verbose=True, model=STASHmaster.ACCESS_ESM1p5.value)


model_drivers = {
    "esm1p5":  Esm1p5Driver,
    "esm1p6":  Esm1p6Driver
}

# Keep track of all um2nc subcommands
all_subcommands = tuple(subparsers.choices.keys())


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


def parse_args():
    # Allow for the 'convert' command to be ommited by adding it if missing.
    try:
        args = parser.parse_args(args=(sys.argv[1:] or ['--help']))
    except argparse.ArgumentError as e:
        if str(e).startswith('argument command: invalid choice:'):
            arg_str = " ".join(sys.argv[1:])
            warnings.warn(f"No command recognised among {all_subcommands}. Running `um2nc convert {arg_str}`")
            args = parser.parse_args(['convert'] + sys.argv[1:])
        else:
            raise e

    return args


def run_command(args):
    # Run selected command
    if args.command == "convert":
        if args.one_nc_per_stash_variable:
            # Single variable files need a DelayedCubePath to resolve the
            # variable name for the output file once it is known
            outfile = DelayedCubePath(args.outfile)
        else:
            outfile = Path(args.outfile)

        process(args.infile, outfile, args)
    elif args.command == "driver":
        driver = model_drivers[args.model_driver](Path(args.model_directory), args.one_nc_per_stash_variable)
        driver.run_conversion(args.delete_ff, args)


def main():
    args = parse_args()
    # Setup logging
    setup_logging(args.verbose, args.quiet, args.strict)

    run_command(args)
