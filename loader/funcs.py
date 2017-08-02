import numpy as np
import theano

import loader.libs.utils as utils
from loader.libs.alphabet import Alphabet

logger = utils.get_logger("LoadData")
MAX_LENGTH = 140

def load_dataset_sequence_labeling(train_path, test_path, word_column=0, target_column=1, label_column=2,
                                   label_name='senti', oov='embedding', embedding="word2Vec",embedding_path=None):

    def construct_tensor(word_index_sentences, target_index_sentences, label_index_sentences):

        X = np.empty([len(word_index_sentences), max_length], dtype=np.int32)
        # T = np.empty([len(target_index_sentences), max_length], dtype=np.int32)
        # Y = []
        Y = np.empty([len(target_index_sentences), 1], dtype=np.int32)
        mask = np.zeros([len(word_index_sentences), max_length], dtype=theano.config.floatX)

        for i in range(len(word_index_sentences)):
            word_ids = word_index_sentences[i]
            # target_ids = target_index_sentences[i]
            label_ids = label_index_sentences[i]
            length = len(word_ids)
            # target_length = len(target_ids)
            for j in range(length):
                wid = word_ids[j]
                X[i, j] = wid

            # for j in range(target_length):
            #     tid = target_ids[j]
            #     T[i, j] = tid

            label = label_ids[0]
            # Y.append(label)
            Y[i, 0] = label
            # Zero out X after the end of the sequence
            X[i, length:] = 0
            # T[i, target_length:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, :length] = 1
        return X, target_index_sentences, Y, mask


    def generate_dataset_fine_tune():

        embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict(embedding, embedding_path, word_alphabet,
                                                                           logger)
        logger.info("Dimension of embedding is %d, Caseless: %d" % (embedd_dim, caseless))
        X_train, T_train, Y_train, mask_train = construct_tensor(word_index_sentences_train, target_index_sentences_train, label_index_sentences_train)
        X_test, T_test, Y_test, mask_test = construct_tensor(word_index_sentences_test, target_index_sentences_test, label_index_sentences_test)

        return X_train, T_train, Y_train, mask_train, X_test, T_test, Y_test, mask_test, \
               build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless), label_alphabet

    word_alphabet = Alphabet('word')
    label_alphabet = Alphabet(label_name)

    # read training data
    logger.info("Reading data from training set...")
    word_sentences_train, _, target_sentences_train, word_index_sentences_train, label_index_sentences_train, target_index_sentences_train = read_sequence(
        train_path, word_alphabet, label_alphabet, word_column, label_column, target_column)

    if oov == "random":
        logger.info("Close word alphabet.")
        word_alphabet.close()

    # read test data
    logger.info("Reading data from test set...")
    word_sentences_test, _, target_sentences_test, word_index_sentences_test, label_index_sentences_test, target_index_sentences_test = read_sequence(
        test_path, word_alphabet, label_alphabet, word_column, label_column, target_column)

    # close alphabets
    word_alphabet.close()
    label_alphabet.close()

    logger.info("word alphabet size: %d" % (word_alphabet.size() - 1))
    logger.info("label alphabet size: %d" % (label_alphabet.size() - 1))

    # get maximum length
    max_length_train = get_max_length(word_sentences_train)
    max_length_test = get_max_length(word_sentences_test)
    max_length = min(MAX_LENGTH, max(max_length_train, max_length_test))
    logger.info("Maximum length of training set is %d" % max_length_train)
    logger.info("Maximum length of test set is %d" % max_length_test)
    logger.info("Maximum length used for training is %d" % max_length)

    logger.info("Generating data with fine tuning...")
    return generate_dataset_fine_tune()


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([word_alphabet.size(), embedd_dim], dtype=theano.config.floatX)
    embedd_table[word_alphabet.default_index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.iteritems():
        ww = word.lower() if caseless else word
        embedd = embedd_dict[ww] if ww in embedd_dict else np.random.uniform(-scale, scale, [1, embedd_dim])
        embedd_table[index, :] = embedd
    return embedd_table

def get_max_length(word_sentences):
    max_len = 0
    for sentence in word_sentences:
        length = len(sentence)
        if length > max_len:
            max_len = length
    return max_len

def read_sequence(path, word_alphabet, label_alphabet, word_column=0, label_column=2, target_column=1):

    word_sentences = []
    label_sentences = []
    target_sentences = []

    word_index_sentences = []
    label_index_sentences = []
    target_index_sentences = []

    num_tokens = 0
    with open(path) as file:
        for line in file:

            words = []
            labels = []
            targets = []
            word_ids = []
            label_ids = []
            target_ids = []

            line.decode('utf-8')
            if line.strip() == "":
                if 0 < len(words) <= MAX_LENGTH:
                    word_sentences.append(words[:])
                    label_sentences.append(labels[:])
                    target_sentences.append(targets[:])

                    word_index_sentences.append(word_ids[:])
                    label_index_sentences.append(label_ids[:])
                    target_index_sentences.append((target_ids[:]))
                    num_tokens += len(words)
                else:
                    if len(words) != 0:
                        logger.info("ignore sentence with length %d" % (len(words)))

                words = []
                labels = []
                targets = []
                word_ids = []
                label_ids = []
                target_ids = []
            else:
                parts = line.strip().split('\t')
                if len(parts) is 3:
                    tokens = parts[word_column].strip().split()
                    label = parts[label_column]
                    target_tokens = parts[target_column].strip().split()
                    for word in tokens:
                        if word != "$t$":
                            words.append(word)  # insert str word into list
                            word_id = word_alphabet.get_index(word)
                            word_ids.append(word_id)
                    for target in target_tokens:
                        targets.append(target)
                        target_id = word_alphabet.get_index(target)
                        target_ids.append(target_id)
                labels.append(label)
                label_id = label_alphabet.get_index(label)
                label_ids.append(label_id)

            if 0 < len(words) <= MAX_LENGTH:
                word_sentences.append(words[:])
                label_sentences.append(labels[:])
                target_sentences.append(targets[:])
                target_sentences.append(targets[:])

                word_index_sentences.append(word_ids[:])
                label_index_sentences.append(label_ids[:])
                target_index_sentences.append((target_ids[:]))
                num_tokens += len(words)
            else:
                if len(words) != 0:
                    logger.info("ignore sentence with length %d" % (len(words)))

    logger.info("#sentences: %d, #tokens: %d" % (len(word_sentences), num_tokens))
    return word_sentences, label_sentences, target_sentences, word_index_sentences, label_index_sentences, target_index_sentences

def build_aspect_embeddings(aspect_index_train, aspect_index_test, embedd_table):

    _, embedd_dim = embedd_table.shape

    def get_target_vec(target_ids):
        targetVec = np.zeros([embedd_dim])
        for id in target_ids:
            xVec = embedd_table[id]
            for i,v in enumerate(xVec):
                targetVec[i] = targetVec[i] + xVec[i]
        for i,v in enumerate(targetVec):
            targetVec[i] = targetVec[i]/ len(target_ids)
        return targetVec

    target_vecs_train = np.array([get_target_vec(target_ids) for target_ids in aspect_index_train])
    target_vecs_test = np.array([get_target_vec(target_ids) for target_ids in aspect_index_test])

    return target_vecs_train, target_vecs_test