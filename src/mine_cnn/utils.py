import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Rotation Invariance CNN')

    parser.add_argument("--epoch", default=100, type=int)

    args = parser.parse_args()
    return args