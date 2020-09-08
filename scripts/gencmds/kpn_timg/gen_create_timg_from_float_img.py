""" scripts.gencmds.kpn_timg.gen_create_timg_from_float_img
    Generate create_timg_from_img commands
"""
import json
import scripts.gencmds.common as common

def _create_parser():
    help_str = 'Create a set of commands to create timgs and logtimgs'
    return common.parser(help_str)

def main(args):
    metadata = json.load(args.input)
    img_md = metadata['image-metadata']
    cmds = []
    for md in img_md:
        pos_args = [md['grayscale-path']]
        for timg_params in md['timgs']:
            cmd = ['python', '-m', 'scripts.base.create_timg_from_float_img']
            cmd.append(md['grayscale-path'])
            if metadata['tmin'] is not None:
                cmd.append('--tmin {}'.format(metadata['tmin']))
            if metadata['tmax'] is not None:
                cmd.append('--tmax {}'.format(metadata['tmax']))
            if metadata['num-avg'] is not None:
                cmd.append('--num-avg {}'.format(metadata['num-avg']))
            if metadata['avg-fn'] is not None:
                cmd.append('--avg-fn {}'.format(metadata['avg-fn']))
            # These should not be None
            cmd.append('--timg {}'.format(timg_params['timg']))
            cmd.append('--logtimg {}'.format(timg_params['logtimg']))
            if timg_params['rng-seed'] is not None:
                cmd.append('--rng-seed {}'.format(timg_params['rng-seed']))
            cmds.append(' '.join(cmd))
    args.output.writelines('\n'.join(cmds))
    return

if __name__ == '__main__':
    main(_create_parser().parse_args())
