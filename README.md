
# shadow_removal_
<img width="3765" height="1880" alt="그림1" src="https://github.com/user-attachments/assets/69f73733-9286-4c9d-8b63-3e8b3a3f8b8f" />
사전 학습 Foundatoin 모델과 Dilated Neighoborhood Attention Transformer(DiNAT)을 기반으로 한 Shadow Removal 딥러닝 모델 설계 

DiNAT의 국소지역 어텐션을 기반으로 그림자 구역 주변에 대한 어텐션을 통한 영역 복원

## Environment
- **OS:** Ubuntu 20.04  
- **Python:** 3.10.18  
- **PyTorch:** 2.3.1  
- **CUDA:** 12.4  
- **GPU:** RTX 3090 24GB x 2
 
---
## training
1. **기본 가중치 다운로드**  
[heira_l_adapter.pth](차후 링크 첨부) 를 다운로드 후 `sam_weight/` 폴더에 저장.

```bash
sam_weight
 └─ heira_l_adapter.pth
```
2. 학습
```bash
python train.py
```
## inference

```bash
python test.py
```
## Result
<img width="741" height="163" alt="shadow_removal" src="https://github.com/user-attachments/assets/d3c9dfe2-7d3e-4705-be2e-0e88a347af8d" />
