import math
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import multiprocessing


class DatasetLoader:
    @staticmethod
    def parse(example):
        features_description = {
            'FileName': tf.io.VarLenFeature(tf.string),
            'FieldDeclarations': tf.io.VarLenFeature(tf.int64),
            'MethodDeclarations': tf.io.VarLenFeature(tf.int64),
            'Ast': tf.io.VarLenFeature(tf.string),
            'Sequence_abs': tf.io.VarLenFeature(tf.int64),
            'Identifiers': tf.io.VarLenFeature(tf.float32),
            'Metrics': tf.io.VarLenFeature(tf.float32),
            'y': tf.io.FixedLenFeature([1], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example, features_description)
        return (tf.sparse.to_dense(parsed_features['FileName'])[0],
                tf.sparse.to_dense(parsed_features['FieldDeclarations']),
                tf.sparse.to_dense(parsed_features['MethodDeclarations']),
                tf.sparse.to_dense(parsed_features['Ast'])[0],
                tf.sparse.to_dense(parsed_features['Sequence_abs']),
                tf.sparse.to_dense(parsed_features['Identifiers']),
                tf.sparse.to_dense(parsed_features['Metrics']),
                parsed_features['y'])

    @staticmethod
    def getNextIteration(datasetFilePath, numberOfEpochs, batchSize, sizeOfTrainData):
        num_batch = math.ceil(sizeOfTrainData * numberOfEpochs / batchSize)
        train_dataset = tf.data.TFRecordDataset(datasetFilePath) \
            .map(DatasetLoader.parse, num_parallel_calls=multiprocessing.cpu_count()) \
            .repeat(numberOfEpochs).shuffle(1024) \
            .padded_batch(batchSize, ) \
            .prefetch(10)
        return tf.python.data.make_one_shot_iterator(train_dataset), num_batch
