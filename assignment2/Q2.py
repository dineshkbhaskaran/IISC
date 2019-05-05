import argparse
from utils import neural_networks

def run_q2(mode):
    print ('Running for mode ' + mode)

    nn = neural_networks(mode)
    nn.load_data('mnist')

    for bs in [16, 32, 128, 1024]:
        for lr in [0.001, 0.01, 0.05, 0.1]:
            stats = nn.fit(lr, bs)
            print (stats)
 
def main():
    run_q2('DNN')
    run_q2('CNN_2D')

if __name__ == '__main__':
    main()
