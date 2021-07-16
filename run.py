from utils import *
from model import model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

bert_path = 'saved_model/kobert.pth'
model.load_state_dict(torch.load(bert_path,map_location=torch.device('cpu') ),strict=False)

def inference(model,sen):
    sen = [sen]
    score = test_sentences(model,sen)
    result = np.argmax(score)
    if result == 1:
        print(f'긍정 문장입니다')
    else:
        print(f'부정 문장입니다')

input_sentence_from_user = input('이 모델은 네이버 영화 댓글 데이터셋으로 학습되었습니다. 문장을 입력해주세요: ')

#실행
inference(model,input_sentence_from_user)
