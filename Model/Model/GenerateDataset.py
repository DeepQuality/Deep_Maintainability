from collections import defaultdict
from glob import glob

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tqdm import tqdm


class Node:
    def __init__(self, label="", parent=None, children=None, num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


def load_dic(path, length):
    # name (SimpleName)	0 5944380
    index = 0
    dic = defaultdict(int)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if index == length:
                break
            dic[line[0]] = int(line[1])
            index += 1
    return dic


def load_metric(path):
    dic = defaultdict(list)
    with open(path, "r", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            if line[2] == 'class':
                filename = line[0] + "," + line[1]
                metrics = line[3:]
                metrics = [-1. if m == 'NaN' else float(m) for m in metrics]
                dic[filename] = metrics
    return dic


def isNum(string):
    return re.compile(r'^\d+$').search(string) is not None


def depth_split(root, depth=0):
    res = defaultdict(list)
    res[depth].append(root)
    for child in root.children:
        for k, v in depth_split(child, depth + 1).items():
            res[k] += v
    return res


def parseTrees(astPath, lines, method_num_threshold, ast_dic):
    trees = []
    line_indexes = []
    for i in range(len(lines)):
        if isNum(lines[i]):
            line_indexes.append(i)
    # trick
    line_indexes.append(len(lines))
    temp_list = []
    for i in range(len(line_indexes) - 1):
        ast_lines = lines[line_indexes[i]:line_indexes[i + 1]]
        temp_list.append(ast_lines)
    if len(temp_list) > method_num_threshold:
        temp_list = random.sample(temp_list, method_num_threshold)

    for ast_lines in temp_list:
        num_nodes = int(ast_lines[0])
        nodes = [Node(num=i, children=[]) for i in range(int(num_nodes))]
        for i in range(num_nodes):
            line = ast_lines[i + 1]
            key = line[line.find(' ') + 1:]

            if key[:len('identifier=')] == "identifier=":
                key = 'identifier'
            elif key[:len('value=')] == "value=":
                key = 'value'
            elif key[:len('content=')] == "content=":
                key = 'content'

            dic_index = ast_dic[key]
            nodes[i].label = dic_index

        for line in ast_lines[num_nodes + 1:]:
            temp = [int(t) for t in line.split(' ')]
            p = temp[0]
            c = temp[1]
            nodes[p].children.append(nodes[c])
            nodes[c].parent = nodes[p]
        depth_split_seq = depth_split(nodes[0])
        trees.append(depth_split_seq)

    pickle.dump(trees, open(astPath, "wb"))


def parse_encodingsOfIdentifier(identifier):
    features = []
    num_part = len(identifier['partNames'])
    for j in range(num_part):
        name = identifier['name']
        mytype = identifier['type']
        numOfParts = identifier['numOfParts']
        partName = identifier['partNames'][j]
        partType = identifier['partTypes'][j]
        isInDic = identifier['isInDics'][j]
        containNum = identifier['containNum']
        containUnderscore = identifier['containUnderscore']
        containDollar = identifier['containDollar']
        feature = [name, mytype, numOfParts, partName, partType, isInDic, containNum, containUnderscore, containDollar, j]
        features.append(feature)
    return features


def convert_field_feature(feature, parts_dic):
    modifier_dic = {'public': 1, 'protected': 2, 'default': 3, 'private': 4}
    variableNameTypeDic = {'class': 1, 'field': 2, 'method': 3, 'parameter': 4, 'variable': 5}
    boolean_dic = {False: 1, True: 2}
    modifier = modifier_dic[feature[0]]
    isStatic = boolean_dic[feature[1]]
    one_field_type = variableNameTypeDic[feature[4]]
    numOfParts = feature[5]
    partName = parts_dic[feature[6]]
    partType = int(feature[7])
    isInDic = boolean_dic[feature[8]]
    containNum = boolean_dic[feature[9]]
    containUnderscore = boolean_dic[feature[10]]
    containDollar = boolean_dic[feature[11]]
    rankPart = feature[12]
    isAssigned = boolean_dic[feature[13]]
    hasComment = boolean_dic[feature[14]]
    numOfVars = feature[15]
    return [partName, modifier, isStatic, one_field_type, numOfParts,
            partType, isInDic, containNum,
            containUnderscore, containDollar,
            isAssigned, hasComment, numOfVars, rankPart]


def parse_field_declarations(line, field_num_threshold, parts_dic):
    try:
        json_data = json.loads(line)
    except:
        json_data = json.loads(line.replace('\\', '\\\\'))
    field_declarations = json_data['FieldDeclarations']
    if len(field_declarations) > field_num_threshold:
        field_declarations = random.sample(field_declarations, field_num_threshold)
    features = []
    for field_declaration in field_declarations:
        num_field = len(field_declaration['encodingsOfField'])
        for i in range(num_field):
            modifier = field_declaration['modifier']
            isStatic = field_declaration['isStatic']
            field_type = field_declaration['type']
            encodingsOfField = field_declaration['encodingsOfField'][i]
            isAssinged = field_declaration['isAssingeds'][i]
            hasComment = field_declaration['hasComment']
            numOfVars = field_declaration['numOfVars']

            one_encodingsOfParameterName = parse_encodingsOfIdentifier(encodingsOfField)
            for temp in one_encodingsOfParameterName:
                features.append([modifier, isStatic, field_type] + temp + [
                    isAssinged, hasComment, numOfVars])
    res = [len(features)]
    for feature in features:
        res += convert_field_feature(feature, parts_dic)
    return res


def convert_method_feature(feature, parts_dic):
    modifier_dic = {'public': 1, 'protected': 2, 'default': 3, 'private': 4}
    variableNameTypeDic = {'class': 1, 'field': 2, 'method': 3, 'parameter': 4, 'variable': 5}
    boolean_dic = {False: 1, True: 2}
    modifier = modifier_dic[feature[0]]
    isStatic = boolean_dic[feature[1]]
    type_method_name = variableNameTypeDic[feature[4]]
    numOfParts_method = feature[5]
    partName_method = parts_dic[feature[6]]
    part_type = int(feature[7])
    part_is_in_dic = boolean_dic[feature[8]]
    containNum_method = boolean_dic[feature[9]]
    containUnderscore_method = boolean_dic[feature[10]]
    containDollar_method = boolean_dic[feature[11]]
    rankPart_method = feature[12]
    one_field_type = variableNameTypeDic[feature[15]] if feature[15] != 0 else 0
    numOfParts_parameter = int(feature[16]) if feature[16] != 0 else 0
    partName_parameter = parts_dic[feature[17]]
    partType = int(feature[18]) if feature[18] != 0 else 0
    isInDic = boolean_dic[feature[19]] if feature[19] != 0 else 0
    containNum = boolean_dic[feature[20]] if feature[20] != 0 else 0
    containUnderscore = boolean_dic[feature[21]] if feature[21] != 0 else 0
    containDollar = boolean_dic[feature[22]] if feature[22] != 0 else 0
    rankPart_parameter = feature[23]
    hasComment = boolean_dic[feature[24]]
    numParameters = feature[25]
    return [partName_method, partName_parameter, modifier, isStatic,
            type_method_name, numOfParts_method,
            part_type, part_is_in_dic,
            containNum_method, containUnderscore_method,
            containDollar_method,
            one_field_type,
            numOfParts_parameter,
            partType, isInDic, containNum,
            containUnderscore, containDollar,
            hasComment, rankPart_method, rankPart_parameter, numParameters]


def parse_method_declarations(line, method_num_threshold, parts_dic):
    try:
        json_data = json.loads(line)
    except:
        json_data = json.loads(line.replace('\\', '\\\\'))
    method_declarations = json_data['MethodDeclarations']
    if len(method_declarations) > method_num_threshold:
        method_declarations = random.sample(method_declarations, method_num_threshold)
    features = []
    for method_declaration in method_declarations:
        modifier = method_declaration['modifier']
        isStatic = method_declaration['isStatic']
        mytype = method_declaration['type']
        hasComment = method_declaration['hasComment']
        numOfParmeters = len(method_declaration['parameterTypes'])

        identifiers_features = parse_encodingsOfIdentifier(method_declaration['encodingOfMethodName'])
        for identifiers_feature in identifiers_features:
            if numOfParmeters == 0:
                feature = [modifier, isStatic, mytype] + identifiers_feature + [0] + [0] * 10 + [hasComment] + [numOfParmeters]
                features.append(feature)
            else:
                for j in range(numOfParmeters):
                    parameterType = method_declaration['parameterTypes'][j]
                    one_encodingsOfParameterName = method_declaration['encodingsOfParameterName'][j]
                    one_encodingsOfParameterName = parse_encodingsOfIdentifier(one_encodingsOfParameterName)
                    for temp in one_encodingsOfParameterName:
                        feature = [modifier, isStatic, mytype] + identifiers_feature + [parameterType] + temp + [hasComment] + [numOfParmeters]
                        features.append(feature)

    res = [len(features)]
    for feature in features:
        res += convert_method_feature(feature, parts_dic)
    return res


# [name, one_field_type, numOfParts, partName, partType, isInDic, containNum, containUnderscore, containDollar]
def convert_identifier_feature(feature, parts_dic):
    boolean_dic = {False: 1, True: 2}
    numOfParts = feature[2]
    partName = parts_dic[feature[3]]
    partType = int(feature[4])
    isInDic = boolean_dic[feature[5]]
    containNum = boolean_dic[feature[6]]
    containUnderscore = boolean_dic[feature[7]]
    containDollar = boolean_dic[feature[8]]
    rankPart = feature[9]
    return [partName, numOfParts, partType, isInDic, containNum, containUnderscore, containDollar, rankPart]


# {"name":"King","type":"Identifier","numOfParts":1,"partNames":["King"],
# "partTypes":["2"],"isInDics":[true],"containNum":false,"containUnderscore":false,"containDollar":false}
def parse_identifiers(line, identifier_num_threshold, parts_dic):
    try:
        json_data = json.loads(line)
    except:
        json_data = json.loads(line.replace('\\', '\\\\'))
    identifiers = json_data['Identifiers']
    if len(identifiers) > identifier_num_threshold:
        identifiers = random.sample(identifiers, identifier_num_threshold)
    features = []
    for identifier in identifiers:
        features += parse_encodingsOfIdentifier(identifier)
    res = [len(features)]
    for feature in features:
        res += convert_identifier_feature(feature, parts_dic)
    return res


def parseOneFile(path, ast_save_dir, abs_tokens_dic,
                 file2metric, sequence_length_threshold,
                 method_num_threshold, ast_dic,
                 field_num_threshold,
                 identifier_num_threshold, partDic, isSingleFile=False):
    with open(path, "r") as f:
        line = f.readline().strip()
        assert line == '-----MetaData-----'
        line = f.readline().strip()
        file_name = line.split("\t")[0] + ',' + line.split("\t")[1]

        # AST
        lines = []
        while True:
            line = f.readline().strip()
            if line == '-----Code Sequence With Abstract-----':
                break
            lines.append(line)
        parseTrees(astPath=os.path.join(ast_save_dir, path.split("/")[-1]), lines=lines,
                   method_num_threshold=method_num_threshold, ast_dic=ast_dic)

        # Code Sequence With Abstract
        lines = []
        while True:
            line = f.readline().strip()
            if line == '-----Field Declarations-----':
                break
            lines.append(line)
        if len(lines) > sequence_length_threshold:
            lines = lines[:sequence_length_threshold]
        tokens_with_abs = [abs_tokens_dic[line] for line in lines]

        # Field Declarations
        line = f.readline().strip()
        field_declaration_indexes = parse_field_declarations(line=line, field_num_threshold=field_num_threshold, parts_dic=partDic)

        # Method Declarations
        temp = f.readline().strip()
        assert temp == '-----Method Declarations-----'
        line = f.readline().strip()
        method_declaration_indexes = parse_method_declarations(line=line, method_num_threshold=method_num_threshold, parts_dic=partDic)

        # Identifiers
        temp = f.readline().strip()
        assert temp == '-----Identifiers-----'
        line = f.readline().strip()
        identifiers = parse_identifiers(line=line, identifier_num_threshold=identifier_num_threshold, parts_dic=partDic)

        # metrics
        if isSingleFile:
            line = f.readline().strip()
            metrics = [float(v) for v in line.split(",")]
            return ([file_name.split(',')[0].encode()], field_declaration_indexes, method_declaration_indexes, [os.path.join(ast_save_dir, path.split("/")[-1]).encode()],tokens_with_abs,identifiers,metrics,0)
        else:
            metrics = file2metric[file_name]
            example = tf.train.Example(features=tf.train.Features(feature={
                'FileName': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name.split(',')[0].encode()])),
                'FieldDeclarations': tf.train.Feature(int64_list=tf.train.Int64List(value=field_declaration_indexes)),
                'MethodDeclarations': tf.train.Feature(int64_list=tf.train.Int64List(value=method_declaration_indexes)),
                'Ast': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.join(ast_save_dir, path.split("/")[-1]).encode()])),
                'Sequence_abs': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens_with_abs)),
                'Identifiers': tf.train.Feature(float_list=tf.train.FloatList(value=identifiers)),
                'Metrics': tf.train.Feature(float_list=tf.train.FloatList(value=metrics)),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if path[-1] == 'h' else 0]))
            }))
            return example


def main(parser):
    addParameters(parser)
    args=parser.parse_args()
    abs_tokens_dic = load_dic(path=args.absDic, length=float('inf'))
    ast_dic = load_dic(path=args.astDic, length=float('inf'))
    parts_dic = load_dic(path=args.partDic, length=args.partNumberDicThreshold)

    file2metric = load_metric(path=args.lowMetricFile)
    file2metric.update(load_metric(path=args.highMetricFile))

    writer = tf.io.TFRecordWriter(path=args.tfrecordsName)
    file_list = glob(pathname=args.trainDir + "*")
    for file in tqdm(iterable=file_list):
        example = parseOneFile(path=file,
                               ast_save_dir=args.astDir,
                               abs_tokens_dic=abs_tokens_dic,
                               file2metric=file2metric,
                               sequence_length_threshold=args.sequenceLengthThreshold,
                               method_num_threshold=args.methodNumberThreshold,
                               ast_dic=ast_dic,
                               field_num_threshold=args.fieldNumberThreshold,
                               identifier_num_threshold=args.identifierNumberThreshold,
                               partDic=parts_dic)
        writer.write(record=example.SerializeToString())
    writer.close()


def parseSingleClass(parser):
    addParameters(parser)
    args=parser.parse_args()
    abs_tokens_dic = load_dic(path=args.absDic, length=float('inf'))
    ast_dic = load_dic(path=args.astDic, length=float('inf'))
    parts_dic = load_dic(path=args.partDic, length=args.partNumberDicThreshold)

    example = parseOneFile(path=args.feature_file,
                           ast_save_dir=args.astDir,
                           abs_tokens_dic=abs_tokens_dic,
                           file2metric=None,
                           sequence_length_threshold=args.sequenceLengthThreshold,
                           method_num_threshold=args.methodNumberThreshold,
                           ast_dic=ast_dic,
                           field_num_threshold=args.fieldNumberThreshold,
                           identifier_num_threshold=args.identifierNumberThreshold,
                           partDic=parts_dic,
                           isSingleFile=True)
    return example


def addParameters(parser):
    parser.add_argument("--astDir", type=str, required=False, default="./temp/",
                        help='Directory saving asts')
    parser.add_argument("--absDic", type=str, required=False, default="./abs_tokens.dic",
                        help='Directory saving asts')
    parser.add_argument("--astDic", type=str, required=False, default="./ast.dic",
                        help='Directory saving asts')
    parser.add_argument("--partDic", type=str, required=False, default="./parts.dic",
                        help='Directory saving asts')

    # stat_2 99 99 99
    parser.add_argument("--fieldNumberThreshold", type=int, required=False, default=35, help='Field Number Threshold')
    parser.add_argument("--methodNumberThreshold", type=int, required=False, default=53, help='Method Number Threshold')
    parser.add_argument("--identifierNumberThreshold", type=int, required=False, default=1376,
                        help='Identifier Number Threshold')
    # stat_1 80 99
    parser.add_argument("--sequenceLengthThreshold", type=int, required=False, default=1647,
                        help='Sequence Length Threshold')
    parser.add_argument("--partNumberDicThreshold", type=int, required=False, default=56949, help='Part Number Dic Threshold')

if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainDir", type=str, required=False, default="/home/yamin/ClassQuality/train/",
                        help='Directory of train or validation or test')
    parser.add_argument("--tfrecordsName", type=str, required=False, default="/home/yamin/ClassQuality/dataset/train.tfrecords",
                        help='Tfrecords name of train or validation or test')
    parser.add_argument("--lowMetricFile", type=str, required=False, default='/home/yamin/ClassQuality/metric_l_class.csv',
                        help='Part Number Threshold')
    parser.add_argument("--highMetricFile", type=str, required=False, default='/home/yamin/ClassQuality/metric_h_class.csv',
                        help='Part Number Threshold')

    main(parser)
