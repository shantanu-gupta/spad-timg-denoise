""" scripts.gencmds.common
"""

import argparse
import sys

def parser(description):
    help_str = description
    parser = argparse.ArgumentParser(description=help_str)
    help_str = 'Input metadata file'
    parser.add_argument('input', nargs='?', type=argparse.FileType('r'),
                        help=help_str,
                        default=sys.stdin)
    help_str = 'Output file (contains a bunch of commands)'
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'),
                        help=help_str,
                        default=sys.stdout)
    return parser

