import os
import sys
import torch
import torch.nn as nn

# src 폴더를 sys.path에 추가 (main_abl.py와 ipynb 모두 호환)
_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

class ModelStrategy:
    """모델별 특수 로직을 캡슐화하는 베이스 클래스"""
    
    def __init__(self):
        self.epoch = 0 # AdaDF에서 사용
        
    def compute_loss(self, model_output, targets, **kwargs):
        """Loss 계산 (모델마다 다름)"""
        raise NotImplementedError
    
    def get_predictions(self, model_output):
        """예측값 추출 (모델마다 출력 구조가 다름)"""
        raise NotImplementedError
    
    def forward_model(self, model, images):
        """모델 forward (기본 구현)"""
        return model(images)
    
def get_model_strategy(model_name, args, device, num_classes):
    """모델별 전략 객체 생성"""
    if model_name == 'ProposedNet':
        from models.ProposedNet.proposednet_strategy import ProposedNetStrategy
        criterion = torch.nn.CrossEntropyLoss()
        return ProposedNetStrategy(criterion)

    else:
        raise ValueError(f"Unknown model name: {model_name}")