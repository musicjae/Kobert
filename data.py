import pandas as pd
from transformers import BertTokenizer
from hyperparameters import args
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils import custom_padsequence

# 판다스로 데이터셋 불러오기
trainset = pd.read_csv('ratings_train.txt', sep='\t')
testset = pd.read_csv('ratings_test.txt', sep='\t')

def preprocessing_dataset(dataset):
    # 리뷰 문장 추출
    sentences = dataset['document']

    # BERT의 입력 형식에 맞게 변환
    sentences = [str(sentence) for sentence in sentences]

    # 라벨 추출
    labels = dataset['label'].values

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = [custom_padsequence(input_id, max_len=args.max_len) for input_id in input_ids]

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks


def DataLoadify_dataset(input_ids, labels, attention_masks):

    if args.mode == 'train':
        # 훈련셋과 검증셋으로 분리
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                            labels,
                                                                                            random_state=827,
                                                                                            test_size=0.1)

        # 어텐션 마스크를 훈련셋과 검증셋으로 분리
        train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                               input_ids,
                                                               random_state=827,
                                                               test_size=0.1)

        # 데이터를 파이토치의 텐서로 변환
        train_inputs = torch.tensor(train_inputs)
        train_labels = torch.tensor(train_labels)
        train_masks = torch.tensor(train_masks)
        validation_inputs = torch.tensor(validation_inputs)
        validation_labels = torch.tensor(validation_labels)
        validation_masks = torch.tensor(validation_masks)

        # 배치 사이즈
        batch_size = args.batch_size

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        results = (train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader)

        return results

    else:
        # 데이터를 파이토치의 텐서로 변환
        test_inputs = torch.tensor(input_ids)
        test_labels = torch.tensor(labels)
        test_masks = torch.tensor(attention_masks)

        # 배치 사이즈
        batch_size = args.batch_size

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        results = (test_data, test_sampler, test_dataloader)

        return results


if args.mode=='train':
    results = DataLoadify_dataset(*preprocessing_dataset(trainset))
elif args.mode == 'test':
    results = DataLoadify_dataset(*preprocessing_dataset(testset))
else:
    results = 0

