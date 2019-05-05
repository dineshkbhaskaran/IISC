from utils import neural_networks

def run_q1():
    nclasses = 10
    #mode = 'RNN'
    #print ('Running for mode ' + mode)

    #nn = neural_networks(mode, nclasses)
    #nn.execute([128], [.01, 0.1])

    #args = {'dropout' : 0.2, 'normalization' : True}
    #nn = neural_networks(mode, nclasses, args)
    #nn.execute([128], [.01, 0.1])
    
    mode = 'LSTM'
    args = {'layers' : 3 }
    nn = neural_networks(mode, nclasses, args)
    nn.execute([128], [.01, 0.1])

def main():
    run_q1()

if __name__ == '__main__':
    main()
