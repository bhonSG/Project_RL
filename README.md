---

# 📘 강화학습의 기초(GITA401) Project

### 📌 조: 44조  
### 👤 팀원: 강보현 (A70027)
### 📂 GitHub Repository  
🔗 https://github.com/bhonSG/Project_RL.git  

## 📑 PPT 보고서  
🔗 **강화학습 Project_A70027강보현.pptx**  
[[https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016](https://github.com/bhonSG/Project_RL/issues/2)](https://github.com/bhonSG/Project_RL/issues/2#issue-3703491094)

## 📁 업로드 파일  
GitHub Issue에 업로드된 데이터 및 소스 파일:

- **user_seg.csv**  
  https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016  

- **ad_watch_hist.csv**  
  https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016  

- **RLProject_A70027.py**  
  https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016  

# 🚀 실행 방법

### 1. 파일 위치
소스 파일 및 업로드 파일을 **동일한 경로**에 위치  
- 업로드 파일: `user_seg.csv`, `ad_watch_hist.csv`  
- 소스 파일: `RLProject_A70027.py`

### 2. 패키지 설치
```bash
pip install keras matplotlib numpy pandas tensorflow
```

### 3. 실행
```bash
python RLProject_A70027.py
```

### 4. 환경 정보
- Python 3.12.2  
- keras==3.12.0  
- matplotlib==3.10.7  
- numpy==2.3.5  
- pandas==2.3.3  
- tensorflow==2.20.0  

---


# 📊 수행 내역 (전체 로그)


📦 펼쳐보기 / 접기

<img width="1920" height="981" alt="rl_training_results png" src="https://github.com/user-attachments/assets/5aed40a3-d52e-4869-b077-ff0d228f628a" />
<img width="1400" height="600" alt="rl_comparison" src="https://github.com/user-attachments/assets/2c14ffcb-e5fd-426a-a748-604f2530c890" />


```
==================================================
원본 데이터 확인
user_seg_df 컬럼: ['user_id', 'seg_id']
user_seg_df 샘플:
  user_id seg_id
0       1     C0
1       2     C0
2       3     G0
3       4      _
4       5     C0

ad_watch_hist_df 컬럼: ['user_id', 'ad_id', 'ad_brand', 'full_watch_cnt']
ad_watch_hist_df 샘플:
   user_id  ad_id  ad_brand  full_watch_cnt
0        1      1        23               1
1        2      1        23               1
...

==================================================
병합된 데이터 shape: (11582, 5)
병합된 데이터 컬럼: ['user_id','ad_id','ad_brand','full_watch_cnt','seg_id']
...

==================================================
State & Action 정의
세그먼트 수: 12
광고 브랜드/카테고리 수: 101
Segment 매핑: {'A0':0, 'B0':1, ... }
Brand/Category 매핑: {...}

==================================================
Reward 정의 (full_watch_cnt 기반)
완전시청(1) 비율: 74.27%
불완전시청(0) 비율: 25.73%
Reward 분포:
0: 2980
1: 8602

==================================================
변환된 데이터 정보
상태 수: 12
액션 수: 230
총 샘플 수: 11582
...

==================================================
Q-Learning 훈련 로그
Episode 10/400, Reward: 151 ...
...
최종 평균 보상: 0.7550

==================================================
DQN 훈련 로그
Episode 10/100, Reward: 72 ...
...
최종 평균 보상: 0.6300

==================================================
정책 평가
[QL] 액션 매칭률: 9.93%
평균 Reward: 1.0
완전시청 달성률: 100%

[DQN] 액션 매칭률: 1.86%
평균 Reward: 1.0
완전시청 달성률: 100%

==================================================
각 고객 세그먼트별 최적 광고 추천
(Q-Learning / DQN 각각 출력)

==================================================
학습 완료!
```




---
