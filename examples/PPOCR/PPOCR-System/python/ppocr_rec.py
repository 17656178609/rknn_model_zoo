# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import numpy as np
import utils.operators
from utils.rec_postprocess import CTCLabelDecode




# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

REC_INPUT_SHAPE = [48, 320] # h,w
CHARACTER_DICT_PATH= '../model/ppocr_keys_v1.txt'

PRE_PROCESS_CONFIG = [ 
        {
            'NormalizeImage': {
                'std': [1, 1, 1],
                'mean': [0, 0, 0],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }
        ]

POSTPROCESS_CONFIG = {
        'CTCLabelDecode':{
            "character_dict_path": CHARACTER_DICT_PATH,
            "use_space_char": True
            }   
        }



def dumpTensor(value, dtype, filePath):
    f = open(filePath, 'w')
    if dtype in ["int", "int32"]:
        np.savetxt(f, np.array(value).flatten(), fmt='%i')
    else:
        np.savetxt(f, np.array(value).flatten())
    f.close()

def readTensor(filePath, dtype, shape):
    if dtype in ["int", "int32"]:
        return np.loadtxt(filePath, dtype=np.int32).reshape(shape)
    else:
        return np.loadtxt(filePath, dtype=np.float32).reshape(shape)

def dumpResult(feed_dict, output_dict, outputDir):
    # save Info
    jsonDict = {}
    jsonDict['inputs'] = []
    jsonDict['outputs'] = []
    for idx, inputName in enumerate(feed_dict.keys()):
        shape = inputInfos[idx][0]
        dtype = inputInfos[idx][1]
        inp = {}
        inp['name'] = inputName
        inp['shape'] = shape
        inp['dtype'] = dtype
        jsonDict['inputs'].append(inp)
    
    for idx, outputName in enumerate(output_dict.keys()):
        shape = outputInfos[idx][0]
        dtype = outputInfos[idx][1]
        outp = {}
        outp['name'] = outputName
        outp['shape'] = shape
        outp['dtype'] = dtype
        jsonDict['outputs'].append(outp)
    # save input, output info
    jsonString = json.dumps(jsonDict, indent=4)
    print("save input / output info")
    with open('{}/input.json'.format(outputDir), 'w') as f:
        f.write(jsonString)
    # dump output
    print("save output:")
    for idx, (outputName, value) in enumerate(output_dict.items()):
        dtype = outputInfos[idx][1]
        print(outputName)
        filename = outputName.replace('/', '.')
        name = '{}/'.format(outputDir) + filename + '_rknn.txt'
        dumpTensor(value, dtype, name)
    return
class TextRecognizer:
    def __init__(self, args) -> None:
        self.model1, self.model2, self.framework = setup_model(args)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.ctc_postprocess = CTCLabelDecode(**POSTPROCESS_CONFIG['CTCLabelDecode'])

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)

        if self.framework == 'onnx':
            image_input = img['image']
            image_input = image_input.reshape(1, *image_input.shape)
            image_input = image_input.transpose(0, 3, 1, 2)
            img['image'] = image_input
        return img
    
    def run(self, imgs):
        outputs=[]
        for img in imgs:
            img = cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
            model_input = self.preprocess({'image':img})
            # 将image的数据类型转换为np.int16
            # model_input['image'] = model_input['image'].astype(np.int16)
            # print("model_input dtype: ", model_input['image'].dtype)


            output = self.model1.run([model_input['image']])
            # output1 = []
            # output = np.array(output).astype(np.int16)
            # output1.append(output)
            # print("output dtype: ", output[0].dtype)
            output = self.model2.run([output[0]])
            preds = output[0].astype(np.float32)
            output = self.ctc_postprocess(preds)
            outputs.append(output)
        return outputs

def setup_model(args):
    model_path1 = args.rec_model_path1
    model_path2 = args.rec_model_path2
    if model_path1.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model1 = RKNN_model_container(model_path1, args.target, args.device_id)
        model2 = RKNN_model_container(model_path2, args.target, args.device_id)
    elif model_path1.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model1 = ONNX_model_container(model_path1)
        model2 = ONNX_model_container(model_path2)
    else:
        assert False, "{} is not rknn/onnx model".format(model_path1)
    print('Model-{} is {} model, starting val'.format(model_path1, platform))
    return model1, model2, platform