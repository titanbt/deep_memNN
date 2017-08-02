import argparse


def DLNLPOptParser():
    
    parser = argparse.ArgumentParser(\
            description='Default DLNLP opt parser.')

    parser.add_argument('-mode', default='deep_memNN')
    parser.add_argument('-config',default='config/sentence_memNN.cfg', help='config file to set.')

    return parser.parse_args()

