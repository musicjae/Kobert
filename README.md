### Comment

- BERT 모델은 데이터 전처리에 비교적 많은 작업이 필요하다. data.py 에서 그런 작업을 수행한다. 이것에 대해 더 자세히 이해하고자 한다면, [1]과 [3]을 참고하세요.
- model.py의 6번째 줄에서 가져온 모델은 소위 pretrained model이다. 이것은 나의 목적에 적합한 모델이 아니다. 내 목적이 영화 댓글의 감성 분석일 때, 나는 라벨링된 영화 댓글 데이터셋이 필요하다. nsmc는 네이버에서 공개한 라벨링된 영화 댓글 세트이다. 이것을 가지고 pretrained model을 미세조정(finetuning) 시킨다.

 
### Performance

| Color Inference | Train Loss | Test Accuracy(%)|Epochs| 
|-------|-------|-------|-------|
|kobert| 0.64 |58|1|



### How to implement
<h4>1) Install required libraries</h4>  

``` shell  
$pip install -r requirements.txt  
```

<h4>2) How to train  </h4>
- 로컬 환경에서 실행 시, 시간 매우 오래 걸림
``` shell
$python train.py --mode=train
```
<h4>3) How to test  </h4>
- 로컬 환경에서 실행 시, 시간 오래 걸림
``` shell
$python test.py --mode=test
```
 
<h4>4) How to run  </h4>
사용자가 직접 문장을 입력한 뒤 그 문장이 긍정인지, 부정인지 분류하기 (권장)

``` shell
$python test.py --mode=test
```

### Possible Issues
- 버트 모델은 매우 무겁기 때문에 학습 및 추론 시 오랜 시간이 걸릴 수 있다. Mac-M1 기준으로 학습 시간이 약 60 시간 이상 소요될 것으로 예상된다. 나는 모델 학습은 구글 colab 환경에서 수행했다. colab pro 환경 기준으로 모델을 1epoch 만큼 학습 하는 데 약 1시간이 소요 됐다.

  
### References
[1] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).  
[2] https://github.com/musicjae/NLP/blob/master/Bert/bert_sentiment_analysis.ipynb  
[3] http://docs.likejazz.com/bert/