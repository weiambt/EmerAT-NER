from transformers import TFBertModel
huggingface_tag = '../huggingface/Bert/bert-base-chinese'
pretrained_model = TFBertModel.from_pretrained(huggingface_tag, from_pt=True)
print(pretrained_model)