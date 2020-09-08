""" scripts.gencmds.kpn_timg.gen_mkdir
    Generate mkdir commands
"""

import json
import scripts.gencmds.common as common

def _create_parser():
    help_str = 'Create mkdir commands for datadirs'
    return common.parser(help_str)

def main(args):
    cmds = []
    md = json.load(args.input)
    datadir = md['data-dir']
    cmds.append('mkdir -p {}'.format(datadir))
    img_md = md['image-metadata']
    for entry in img_md:
        cmds.append('mkdir -p {}'.format(entry['data-dir']))
        cmds.append('mkdir -p {}'.format(entry['timg-dir']))
        cmds.append('mkdir -p {}'.format(entry['logtimg-dir']))
    args.output.writelines('\n'.join(cmds))

if __name__ == '__main__':
    main(_create_parser().parse_args())
