import argparse
import os
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import subprocess
import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import GenerateDataset
from Model import Model


def parseClass(classPath, parser):
    args = parser.parse_args()
    javacmd = 'java -Xss1g -jar ' + args.parse_tool_path + ' one ' \
              + classPath + ' > ' + args.feature_file
    cmd = subprocess.Popen(javacmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        cmd.wait(20)
        cmd.terminate()
    except Exception as e:
        timeout_str = classPath
        print('TIMEOUT', timeout_str)
    cmd.terminate()


def parseFeature(parser):
    example = GenerateDataset.parseSingleClass(parser)
    identifier = example[5]
    return (example[0], example[1], example[2], example[3], example[4],
            [float(a) for a in identifier], example[6], example[7])


def getScore(example, parser):
    args = parser.parse_args()
    model = Model(args=args)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))
    score = model.predict(example[1:7])
    return score


def main(fileName, parser):
    parseClass(fileName, parser)
    example = parseFeature(parser)
    example = [[a] for a in example]
    example[3] = tf.constant(example[3][0],dtype=tf.string)
    score = getScore(example, parser)
    return score.numpy()[0][0]


if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=False,
                        default='./all/',
                        help='Model directory')
    parser.add_argument("--single_class_path", type=str, required=True, default='./Test.java',
                        help='Please input a class!')
    parser.add_argument("--feature_file", type=str, required=False, default='./features_temp',
                        help='Feature file')
    parser.add_argument("--parse_tool_path", type=str, required=False,
                        default='./JavaParserForClass-1.0-SNAPSHOT-jar-with-dependencies.jar',
                        help='Tool for parsing features')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    args = parser.parse_args()
    score = main(args.single_class_path, parser)
    print(args.single_class_path, score, sep='\t')
