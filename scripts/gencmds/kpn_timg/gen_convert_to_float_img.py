""" scripts.gencmds.kpn_timg.gen_convert_to_float_img
    Generate convert_to_float_img commands
"""

import json
import scripts.gencmds.common as common

def _create_parser():
    help_str = 'Generate a bunch of convert_to_float_img commands'
    return common.parser(help_str)

def main(args):
    metadata = json.load(args.input)
    img_md = metadata['image-metadata']
    cmds = []
    for md in img_md:
        cmd = ['python', '-m', 'scripts.base.convert_to_float_img']
        cmd.append(md['original-copy-path'])
        cmd.append(md['grayscale-path'])
        if metadata['spatial-downscale'] is not None:
            cmd.append('--spatial-downscale {}'.format(
                                        metadata['spatial-downscale']))
        cmds.append(' '.join(cmd))
    args.output.writelines('\n'.join(cmds))
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
