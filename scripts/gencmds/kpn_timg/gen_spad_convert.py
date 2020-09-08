""" scripts.gencmds.kpn_timg.gen_spad_convert
    Generate spad_convert commands to compute ground truth timgs and logtimgs
"""
import json
import scripts.gencmds.common as common

def _create_parser():
    help_str = ('Create a set of commands to create ground truth timgs and '
                'logtimgs.')
    return common.parser(help_str)

def main(args):
    metadata = json.load(args.input)
    img_md = metadata['image-metadata']
    cmds = []
    for md in img_md:
        grayscale = md['grayscale-path']
        timg_path = md['true-timg-path']
        logtimg_path = md['true-logtimg-path']

        cmd = ['python', '-m', 'scripts.base.spad_convert']
        cmd.append(grayscale)
        cmd.append(timg_path)
        cmd.append('--mapping radiance-to-timg')
        if metadata['max-photon-rate'] is not None:
            cmd.append('--max-photon-rate {}'.format(
                                                metadata['max-photon-rate']))
        cmds.append(' '.join(cmd))
        
        cmd = ['python', '-m', 'scripts.base.spad_convert']
        cmd.append(grayscale)
        cmd.append(logtimg_path)
        cmd.append('--mapping radiance-to-logtimg')
        if metadata['max-photon-rate'] is not None:
            cmd.append('--max-photon-rate {}'.format(
                                                metadata['max-photon-rate']))
        cmds.append(' '.join(cmd))
    args.output.writelines('\n'.join(cmds))
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
