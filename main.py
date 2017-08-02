from utils.commandParser import DLNLPOptParser
import warnings
from model.sentence_memNN_main import SentenceMemNNMain

warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':

    args = DLNLPOptParser()
    config = args.config

    if args.mode == 'deep_memNN':
        deep_memNN = SentenceMemNNMain(config, args)
        deep_memNN.run()