from hyperparameters import device
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

# 분류를 위한 BERT 모델 생성
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
model.to(device)