def main(args):
    print(args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g0', nargs='*')
    parser.add_argument('--g1', nargs='*')
    parser.add_argument('--g2', nargs='*')
    parser.add_argument('--g3', nargs='*')
    parser.add_argument('--g4', nargs='*')
    main(parser.parse_args())
