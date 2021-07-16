import argparse
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

parser = argparse.ArgumentParser(description='hyperparameters.....')
parser.add_argument('--max_len',type=int,default=128) # 학습 데이터셋에 들어갈 문장의 최대 길이를 설정해준다. (예) 형태소의 개수가 60개인 샘플의 경우, 128개의 형태소로 이루어진 것처럼 패딩값(0)을 61부터 추가, 형태소 200개의 경우, 128번째 형태소까지만 저장하고 나머지 자름
parser.add_argument('--batch_size',type=int,default=4) # 로컬 환경에서 배치 사이즈가 크면 학습 혹은 추론 시 Out of Memory 발생할 수 있음. 보통 배치 16 아래는 작다고, 그 이상부터는 크다고 표현하는 경향이 있음
parser.add_argument('--epochs',type=int,default=1)
parser.add_argument('--lr',type=float,default=2e-5)
parser.add_argument('--eps',type=float,default=1e-8)
parser.add_argument('--mode',type=str,default='train') # 모델 학습 시, train, 테스트 시, test

args = parser.parse_args()