import argparse
from typing import Optional

__all__ = ["RawTextArgumentDefaultsHelpFormatter", "none_or_str"]


class RawTextArgumentDefaultsHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def none_or_str(value: str) -> Optional[str]:
    return None if value == "None" else value
