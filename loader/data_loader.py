import loader.funcs as funcs

class DataLoader(object):
    def __init__(self, train_file, test_file, word_column, label_column, target_column,
                 oov='embedding', embedding='glove', embedding_path=None):

        self.train_file = train_file
        self.test_file = test_file
        self.word_column = word_column
        self.label_column = label_column
        self.target_column = target_column
        self.embedding = embedding
        self.embedding_path = embedding_path
        self.oov = oov

        self.data = {'X_train': [], 'T_train': [], 'Y_train': [], 'mask_train': [],
                     'X_test': [], 'T_test': [], 'Y_test': [], 'mask_test': [],
                     'embedd_table': [], 'label_alphabet': []
                     }

    def load_data(self):
        self.data['X_train'], self.data['T_train'], self.data['Y_train'], self.data['mask_train'], \
        self.data['X_test'], self.data['T_test'], self.data['Y_test'], self.data['mask_test'], \
        self.data['embedd_table'], self.data['label_alphabet'] = funcs.load_dataset_sequence_labeling(self.train_file,
                                                                              self.test_file,
                                                                              word_column=self.word_column,
                                                                              target_column=self.target_column,
                                                                              label_column=self.label_column,
                                                                              oov=self.oov,
                                                                              embedding=self.embedding,
                                                                              embedding_path=self.embedding_path)
        return self.data