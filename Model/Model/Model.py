import pickle
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Embedding
from TreeLSTM import ChildSumLSTMLayer
from TreeLSTM import tree2tensor
from TreeLSTM import TreeEmbeddingLayer
from TreeLSTM import Node


class Model(tf.keras.Model):
    def __init__(self, args):
        super(Model, self).__init__(name='Class_Quality_Model')
        # get model name
        parts = args.checkpoint_dir.split('/')
        if len(parts[-1]) == 0:
            self.model = parts[-2]
        else:
            self.model = parts[-1]
        # print('Current Model:', self.model)

        self.numberOfStatisticalTermsOfFieldDeclaration = 13
        self.numberOfStatisticalTermsOfMethodHeader = 20
        self.sizeOfVocabularyForSimplifiedASTs = 591 + 1
        self.sizeOfVocabularyForStructuralFormat = 110 + 1
        self.thresholdForLengthOfStructuralFormat = 1647
        self.numberOfStatisticalTermsOfIdentifier = 7
        self.sizeOfVocabularyForSoftWords = 56949 + 1
        self.numberOfTermsOfMetrics = 39

        if 'disableSimplifiedASTs' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, True, False, True, True, True
        elif 'disableIdentifiers' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, True, True, True, False, True
        elif 'disableMethodHeaders' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, False, True, True, True, True
        elif 'disableAbs' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, True, True, False, True, True
        elif 'disableField' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                False, True, True, True, True, True
        elif 'disableMetric' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, True, True, True, True, False
        elif 'all' in self.model:
            self.enableFieldDeclarations, self.enableMethodHeaders, self.enableSimplifiedASTs, \
            self.enableStructuralFormat, self.enableIdentifiers, self.enableMetrics = \
                True, True, True, True, True, True

        if self.enableFieldDeclarations or self.enableMethodHeaders or self.enableIdentifiers:
            self.embeddingLengthOfSoftWord = 30
            self.partName_E = Embedding(self.sizeOfVocabularyForSoftWords, self.embeddingLengthOfSoftWord,
                                        mask_zero=True, name='partName_E')
        if self.enableFieldDeclarations:
            self.field_dense1 = Dense(input_shape=(self.numberOfStatisticalTermsOfFieldDeclaration + self.embeddingLengthOfSoftWord,), units=16,
                                      activation=tf.nn.elu, name='field_dense1')
            self.field_dense2 = Dense(input_shape=(16,), units=8, activation=None, name='field_dense2')
            self.field_dense3 = Dense(input_shape=(8,), units=1, activation=None, name='field_dense3')
        if self.enableMethodHeaders:
            self.method_dense1 = Dense(input_shape=(self.numberOfStatisticalTermsOfMethodHeader + self.embeddingLengthOfSoftWord + self.embeddingLengthOfSoftWord,),
                                       units=16, activation=tf.nn.elu, name='method_dense1')
            self.method_dense2 = Dense(input_shape=(16,), units=8, activation=None, name='method_dense2')
            self.method_dense3 = Dense(input_shape=(8,), units=1, activation=None, name='method_dense3')
        if self.enableSimplifiedASTs:
            self.embeddingLengthOfNode = 8
            self.dimensionOfPresentation = 8
            self.layerOfTreeLSTM = 1
            self.tree_embeddings = TreeEmbeddingLayer(self.embeddingLengthOfNode, self.sizeOfVocabularyForSimplifiedASTs)
            for i in range(self.layerOfTreeLSTM):
                self.__setattr__("layer{}".format(i), ChildSumLSTMLayer(self.embeddingLengthOfNode,
                                                                        self.dimensionOfPresentation,
                                                                        name='ast_TreeLSTM_' + str(i)))
            self.ast_dense1 = Dense(input_shape=(self.dimensionOfPresentation,),
                                    units=16,
                                    activation=tf.nn.elu, name='ast_dense1')
            self.ast_dense2 = Dense(input_shape=(16,), units=8, activation=None, name='ast_dense2')
            self.ast_dense3 = Dense(input_shape=(8,), units=1, activation=None, name='ast_dense3')
        if self.enableStructuralFormat:
            self.embeddingLengthOfToken = 8
            self.sequence_abs_E = Embedding(self.sizeOfVocabularyForStructuralFormat, self.embeddingLengthOfToken,
                                            mask_zero=True, name='sequence_abs_E')
            self.sequence_abs_lstm = LSTM(input_shape=(None, None, self.embeddingLengthOfToken),
                                          units=32, name='sequence_abs_lstm')
        if self.enableIdentifiers:
            self.identifier_dense1 = Dense(input_shape=(self.numberOfStatisticalTermsOfIdentifier + self.embeddingLengthOfSoftWord,),
                                           units=16, activation=tf.nn.elu, name='identifier_dense1')
            self.identifier_dense2 = Dense(input_shape=(16,), units=8, activation=None, name='identifier_dense2')
            self.identifier_dense3 = Dense(input_shape=(8,), units=1, activation=None, name='identifier_dense3')
        if self.enableMetrics:
            self.metric_dense1 = Dense(input_shape=(self.numberOfTermsOfMetrics,),
                                       units=32, activation=tf.nn.elu, name='metric_dense1')
            self.metric_dense2 = Dense(input_shape=(32,), units=16, activation=tf.nn.elu, name='metric_dense2')
            self.metric_dense3 = Dense(input_shape=(16,), units=8, activation=tf.nn.elu, name='metric_dense3')

        self.last_dense1 = Dense(units=32, activation=tf.nn.elu, name='last_dense1')
        self.last_dense2 = Dense(units=16, activation=tf.nn.elu, name='last_dense2')
        self.last_dense3 = Dense(units=8, activation=tf.nn.elu, name='last_dense3')
        self.last_dense4 = Dense(units=1, name='last_dense4')

    def getEmbeddingOfFieldDeclarations(self, x):
        lens = []
        classes = []
        for FieldDeclarations in x:
            if FieldDeclarations[0] == 0:
                FieldDeclarations = tf.zeros([1, 1 + self.numberOfStatisticalTermsOfFieldDeclaration], tf.float32)
                lens.append(1)
            else:
                numField = FieldDeclarations[0]
                lens.append(numField)
                fields = FieldDeclarations[1:numField * (1 + self.numberOfStatisticalTermsOfFieldDeclaration) + 1]
                fields = tf.cast(fields, dtype=tf.float32)
                FieldDeclarations = tf.reshape(fields, [numField, (1 + self.numberOfStatisticalTermsOfFieldDeclaration)])
            classes.append(FieldDeclarations)

        classes = tf.concat(classes, axis=0)
        token_indexes, features = tf.split(classes, [1, self.numberOfStatisticalTermsOfFieldDeclaration], 1)
        aa = self.partName_E(token_indexes)
        aa = tf.squeeze(aa, [1])
        classes = tf.concat([aa, features], 1)

        weights = self.field_dense1(classes)
        weights = self.field_dense2(weights)
        weights = self.field_dense3(weights)
        weights = tf.split(weights, lens, 0)

        total_weights = []
        for weight in weights:
            weight = tf.nn.softmax(tf.squeeze(weight, axis=1))
            total_weights.append(weight)
        total_weights = tf.concat(total_weights, axis=0)
        total_weights = tf.expand_dims(total_weights, 1)
        total = total_weights * classes
        total = tf.split(total, lens, 0)
        tatal_total = []
        for one in total:
            one = tf.reduce_sum(one, 0)
            tatal_total.append(one)
        output = tf.concat(tatal_total, axis=0)
        output = tf.reshape(output, [len(x), self.numberOfStatisticalTermsOfFieldDeclaration + self.embeddingLengthOfSoftWord])
        return output

    def getEmbeddingOfMethodHeaders(self, x):
        lens = []
        classes = []
        for MethodDeclarations in x:
            if MethodDeclarations[0] == 0:
                MethodDeclarations = tf.zeros([1, (2 + self.numberOfStatisticalTermsOfMethodHeader)], tf.float32)
                lens.append(1)
            else:
                numMethods = MethodDeclarations[0]
                lens.append(numMethods)
                methods = MethodDeclarations[1:numMethods * (2 + self.numberOfStatisticalTermsOfMethodHeader) + 1]
                methods = tf.cast(methods, dtype=tf.float32)
                MethodDeclarations = tf.reshape(methods, [numMethods, (2 + self.numberOfStatisticalTermsOfMethodHeader)])
            classes.append(MethodDeclarations)

        classes = tf.concat(classes, axis=0)
        token_indexes_method, token_indexes_parameter, features = tf.split(classes, [1, 1, self.numberOfStatisticalTermsOfMethodHeader], 1)
        aa = self.partName_E(token_indexes_method)
        aa = tf.squeeze(aa,[1])
        bb = self.partName_E(token_indexes_parameter)
        bb = tf.squeeze(bb, [1])
        classes = tf.concat([aa, bb, features], 1)

        weights = self.method_dense1(classes)
        weights = self.method_dense2(weights)
        weights = self.method_dense3(weights)
        weights = tf.split(weights, lens, 0)

        total_weights = []
        for weight in weights:
            weight = tf.nn.softmax(tf.squeeze(weight, axis=1))
            total_weights.append(weight)
        total_weights = tf.concat(total_weights, axis=0)
        total_weights = tf.expand_dims(total_weights, 1)
        total = total_weights * classes
        total = tf.split(total, lens, 0)
        tatal_total = []
        for one in total:
            one = tf.reduce_sum(one, 0)
            tatal_total.append(one)
        output = tf.concat(tatal_total, axis=0)
        output = tf.reshape(output, [len(x), self.numberOfStatisticalTermsOfMethodHeader + self.embeddingLengthOfSoftWord + self.embeddingLengthOfSoftWord])
        return output

    def getEmbeddingOfMetrics(self, x):
        classes = []
        for Metrics in x:
            if len(Metrics) == 0:
                Metrics = tf.zeros([1, self.numberOfTermsOfMetrics], tf.float32)
            else:
                Metrics = tf.reshape(Metrics, [1, self.numberOfTermsOfMetrics])
            classes.append(Metrics)

        classes = tf.concat(classes, axis=0)
        output = self.metric_dense1(classes)
        output = self.metric_dense2(output)
        output = self.metric_dense3(output)
        output = tf.stack(output)
        return output

    def getEmbeddingOfStructuralFormat(self, Sequence_abss):
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            Sequence_abss, padding="post", truncating='post', maxlen=self.thresholdForLengthOfStructuralFormat)
        x = self.sequence_abs_E(padded_inputs)
        mask = self.sequence_abs_E.compute_mask(padded_inputs)
        output = self.sequence_abs_lstm(x, mask=mask)
        return output

    def getEmbeddingOfIdentifiers(self, x):
        lens = []
        classes = []
        for Identifiers in x:
            if Identifiers[0] == 0:
                Identifiers = tf.zeros([1, 1 + self.numberOfStatisticalTermsOfIdentifier], tf.int64)
                lens.append(1)
            else:
                numIdentifiers = Identifiers[0]
                numIdentifiers = tf.cast(numIdentifiers, tf.int32)
                lens.append(numIdentifiers)
                methods = Identifiers[1:numIdentifiers * (1 + self.numberOfStatisticalTermsOfIdentifier) + 1]
                Identifiers = tf.reshape(methods, [numIdentifiers, (1 + self.numberOfStatisticalTermsOfIdentifier)])
            classes.append(Identifiers)

        classes = tf.concat(classes, axis=0)
        token_indexes, features = tf.split(classes, [1, self.numberOfStatisticalTermsOfIdentifier], 1)
        aa = self.partName_E(token_indexes)
        aa = tf.squeeze(aa)
        total_features = tf.concat([aa, features], 1)
        weights = self.identifier_dense1(total_features)
        weights = self.identifier_dense2(weights)
        weights = self.identifier_dense3(weights)
        weights = tf.split(weights, lens, 0)
        total_weights = []
        for weight in weights:
            weight = tf.nn.softmax(tf.squeeze(weight, axis=1))
            total_weights.append(weight)
        total_weights = tf.concat(total_weights, axis=0)
        total_weights = tf.expand_dims(total_weights, 1)
        total = total_weights * total_features
        total = tf.split(total, lens, 0)
        tatal_total = []
        for one in total:
            one = tf.reduce_sum(one, 0)
            tatal_total.append(one)
        output = tf.concat(tatal_total, axis=0)
        output = tf.reshape(output, [len(x), self.numberOfStatisticalTermsOfIdentifier + self.embeddingLengthOfSoftWord])
        return output

    def getEmbeddingOfSimplifiedASTs(self, asts):
        treesFromFile = []
        for i in range(len(asts)):
            with open(asts[i].numpy().decode(), 'rb') as f:
                trees = pickle.load(f)
                treesFromFile.append(trees)
        if len(treesFromFile) == 1 and len(treesFromFile[0]) == 0:
            return tf.zeros([len(asts), self.dimensionOfPresentation], tf.float32)

        rootss = []
        lens = []
        for trees in treesFromFile:
            lens.append(len(trees))
            for tree in trees:
                rootss.append(tree)

        tensor, indice = tree2tensor(rootss)
        tensor = self.tree_embeddings(tensor)
        for i in range(self.layerOfTreeLSTM):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        h = tensor[-1]

        weights = self.ast_dense1(h)
        weights = self.ast_dense2(weights)
        weights = self.ast_dense3(weights)
        weights = tf.split(weights, lens, 0)
        total_weights = []
        for weight in weights:
            weight = tf.nn.softmax(tf.squeeze(weight, axis=1))
            total_weights.append(weight)
        total_weights = tf.concat(total_weights, axis=0)
        total_weights = tf.expand_dims(total_weights, 1)
        total = total_weights * h
        total = tf.split(total, lens, 0)
        tatal_total = []
        for one in total:
            one = tf.reduce_sum(one, 0)
            tatal_total.append(one)
        output = tf.concat(tatal_total, axis=0)
        output = tf.reshape(output, [len(asts), self.dimensionOfPresentation])
        return output

    def call(self, xs):
        output = None
        if self.enableFieldDeclarations:
            output_fields = self.getEmbeddingOfFieldDeclarations(xs[0])
        if self.enableMethodHeaders:
            output_methods = self.getEmbeddingOfMethodHeaders(xs[1])
        if self.enableSimplifiedASTs:
            output_asts = self.getEmbeddingOfSimplifiedASTs(xs[2])
        if self.enableStructuralFormat:
            output_sequence_abs = self.getEmbeddingOfStructuralFormat(xs[3])
        if self.enableIdentifiers:
            output_identifier = self.getEmbeddingOfIdentifiers(xs[4])
        if self.enableMetrics:
            output_metrics = self.getEmbeddingOfMetrics(xs[5])

        if 'all' in self.model:
            output = tf.concat([output_fields, output_methods,output_asts,
                                output_sequence_abs, output_identifier, output_metrics], axis=1)
        elif 'disableAST' == self.model:
            output = tf.concat([output_fields, output_methods,
                                output_sequence_abs, output_identifier, output_metrics], axis=1)
        elif 'disableIdentifier' == self.model:
            output = tf.concat([output_fields, output_methods, output_asts,
                                output_sequence_abs, output_metrics], axis=1)
        elif 'disableMethod' == self.model:
            output = tf.concat([output_fields, output_asts,
                                output_sequence_abs, output_identifier, output_metrics], axis=1)
        elif 'disableAbs' == self.model:
            output = tf.concat([output_fields, output_methods, output_asts,
                                output_identifier, output_metrics], axis=1)
        elif 'disableField' == self.model:
            output = tf.concat([output_methods, output_asts,
                                output_sequence_abs, output_identifier, output_metrics], axis=1)
        elif 'disableMetric' == self.model:
            output = tf.concat([output_fields, output_methods, output_asts,
                                output_sequence_abs, output_identifier], axis=1)

        features = self.last_dense1(output)
        features = self.last_dense2(features)
        features = self.last_dense3(features)
        features = self.last_dense4(features)
        features = tf.nn.sigmoid(features)
        return features

    def predict(self, inputs):
        return self(inputs)
