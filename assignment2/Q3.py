import argparse
from utils import neural_networks

def run_q3(mode):
    print ('Running for mode ' + mode)

    nn = neural_networks(mode)
    nn.load_data('cifar')

    for bs in [128, 256]:
        for lr in [0.01, 0.05]:
            stats = nn.fit(lr, bs)
            print (stats)

def main():
    run_q3('CNN_3D')

if __name__ == '__main__':
    main()
