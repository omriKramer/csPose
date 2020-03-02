import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.script + '_' + '_'.join(args.training_script_args)
    out_file = name + '.out'
    err_file = name + '.err'
    cmd = ['bsub', '-env', 'LSB_CONTAINER_IMAGE=ibdgx001:5000/omri-20.02', '-q', 'waic-long',
           '-gpu', 'num=1:j_exclusive=yes', '-R', 'rusage[mem=8192]', '-R', 'affinity[thread*8]',
           '-app', 'nvidia-gpu', '-oo', out_file, '-eo', err_file, 'python'] + args.training_script_args
    print(cmd)
    subprocess.call(cmd)


if __name__ == '__main__':
    main()
