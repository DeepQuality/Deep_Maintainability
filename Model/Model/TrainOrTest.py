import argparse
import os
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tqdm import tqdm
from DatasetLoader import DatasetLoader
from Model import Model
from TreeLSTM import Node


def train(args, restore=False):
    model = Model(args=args)
    if restore:
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
        checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    log_dir = args.log_dir + time.strftime("%m-%d %H:%M:%S", time.localtime()) + "_" \
                           + str([model.enableFieldDeclarations, model.enableMethodHeaders, model.enableSimplifiedASTs,
                                  model.enableStructuralFormat, model.enableMetrics])

    summary_writer = tf.summary.create_file_writer(log_dir)

    train_iter, num_batch = DatasetLoader.getNextIteration(datasetFilePath=args.trainDataset,
                                                   numberOfEpochs=args.epoch,
                                                   batchSize=args.batchSize,
                                                   sizeOfTrainData=args.num_train)
    bar = tqdm(range(num_batch), ncols=128)
    for batch_index in bar:
        batch = train_iter.get_next()
        with tf.GradientTape() as tape:
            features = model(batch[1:7])
            y_pred = tf.concat([1 - features, features], 1)
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true=tf.squeeze(tf.one_hot(batch[7], 2)), y_pred=y_pred))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            bar.set_description(str(loss.numpy()))
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=batch_index)
            if batch_index == 0:
                model.summary()

        if batch_index != 0 and batch_index % 20000 == 0:
            checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
            checkpoint.save(file_prefix=args.checkpoint_dir + 'cp')
            test(args=args, model=model, datasetFilePath=args.validationDataset, num_example=args.num_validation,
                epoch=int(batch_index / 20000), summary_writer=summary_writer)


def test(args, model, datasetFilePath, num_example,
         toPrintLabel=False, epoch=None, summary_writer=None):
    files = []
    scores = []
    if model is None:
        model = Model(args=args)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
        checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))

    test_iter, num_batch = DatasetLoader.getNextIteration(datasetFilePath=datasetFilePath,
                                                          numberOfEpochs=args.epoch,
                                                          batchSize=args.batchSize,
                                                          sizeOfTrainData=args.num_train)

    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    accuracy1 = tf.keras.metrics.Accuracy()
    precision1 = tf.keras.metrics.Precision()
    recall1 = tf.keras.metrics.Recall()

    for batch in test_iter:
        features = model.predict(batch[1:7])
        if toPrintLabel:
            files += batch[0].numpy().tolist()
            scores += tf.squeeze(features).numpy().tolist()

        y_true = tf.squeeze(batch[7]).numpy()
        y_pred = tf.concat([1 - features, features], 1)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.float32).numpy()
        accuracy.update_state(y_true=y_true, y_pred=y_pred)
        precision.update_state(y_true=y_true, y_pred=y_pred)
        recall.update_state(y_true=y_true, y_pred=y_pred)

        y_true1 = 1 - tf.squeeze(batch[7]).numpy()
        y_pred1 = tf.concat([features, 1 - features], 1)
        y_pred1 = tf.cast(tf.argmax(y_pred1, axis=1), dtype=tf.float32).numpy()
        accuracy1.update_state(y_true=y_true1, y_pred=y_pred1)
        precision1.update_state(y_true=y_true1, y_pred=y_pred1)
        recall1.update_state(y_true=y_true1, y_pred=y_pred1)

    accuracy = accuracy.result().numpy()
    precision = precision.result().numpy()
    recall = recall.result().numpy()

    accuracy1 = accuracy1.result().numpy()
    precision1 = precision1.result().numpy()
    recall1 = recall1.result().numpy()
    if summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar("accuracy", accuracy, step=epoch)
            tf.summary.scalar("precision", precision, step=epoch)
            tf.summary.scalar("recall", recall, step=epoch)

    print("\naccuracy: %f\tprecision: %f\trecall: %f\tF1: %f" % (accuracy, precision, recall, (2*precision*recall/(precision+recall))))
    print("accuracy1: %f\tprecision1: %f\trecall1: %f\tF11: %f\n" % (accuracy1, precision1, recall1,(2*precision1*recall1/(precision1+recall1))))
    if toPrintLabel:
        lines = [files[i].decode() + "\t" + str(scores[i]) + "\n" for i in range(len(files))]
        with open(args.checkpoint_dir + 'score', 'w') as f:
            f.writelines(lines)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    if args.toTrain == "True":
        print("train")
        train(args=args)
    print("test")
    test(args=args, model=None, datasetFilePath=args.testDataset, num_example=args.num_test, toPrintLabel=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=str, required=False,
                        default="0",
                        help='GPU')
    parser.add_argument("--toTrain", type=str, required=False,
                        default="False",
                        help='toTrain')
    parser.add_argument("--astsPath", type=str, required=False,
                        default="/home/yamin/ClassQuality/dataset/asts/",
                        help='astsPath')
    parser.add_argument("--num_train", type=int, required=False, default=1338732,
                        help='num_train')
    parser.add_argument("--num_validation", type=int, required=False, default=27890,
                        help='num_validation')
    parser.add_argument("--num_test", type=int, required=False, default=27892,
                        help='num_test')
    parser.add_argument("--trainDataset", type=str, required=False,
                        default="/home/yamin/ClassQuality/dataset/train.tfrecords",
                        help='trainDataset')
    parser.add_argument("--validationDataset", type=str, required=False,
                        default="/home/yamin/ClassQuality/dataset/validation.tfrecords",
                        help='validationDataset')
    parser.add_argument("--testDataset", type=str, required=False,
                        default="/home/yamin/ClassQuality/dataset/test.tfrecords",
                        help='testDataset')
    parser.add_argument("--epoch", type=int, required=False, default=3,
                        help='epoch')
    parser.add_argument("--batchSize", type=int, required=False, default=32,
                        help='batchSize')
    parser.add_argument("--scorePath", type=str, required=False,
                        default='../ModelDir/all-t3/',
                        help='scorePath')
    parser.add_argument("--learning_rate", type=float, required=False,
                        default=0.001,
                        help='learning_rate')
    parser.add_argument("--checkpoint_dir", type=str, required=False,
                        default='../ModelDir/all-t3/',
                        help='checkpoint_dir')
    parser.add_argument("--log_dir", type=str, required=False,
                        default='../ModelLog/',
                        help='log_dir')

    main(args=parser.parse_args())
