""" scripts.gencmds.kpn_timg.gen_cp_orig
"""

import json
import scripts.gencmds.common as common

def _create_parser():
    help_str = 'Create commands to copy original image files to datadirs'
    return common.parser(help_str)

def main(args):
    cmds = []
    md = json.load(args.input)
    cmds.append('cp {} {}/'.format(args.input.name, md['data-dir']))
    img_md = md['image-metadata']
    for entry in img_md:
        cmds.append('cp {} {}'.format(entry['original-path'],
                                    entry['original-copy-path']))
    args.output.writelines('\n'.join(cmds))

if __name__ == '__main__':
    main(_create_parser().parse_args())

