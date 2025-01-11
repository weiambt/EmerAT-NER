# from transformers import TFBertModel
# huggingface_tag = '../huggingface/Bert/bert-base-chinese'
# pretrained_model = TFBertModel.from_pretrained(huggingface_tag, from_pt=True)
# print(pretrained_model)

import tensorflow as tfv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tfv2.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tfv2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)