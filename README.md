# Project_RL

강화학습의 기초(GITA401) Project

조 : 44조
팀원 : 강보현

Git 주소 : https://github.com/bhonSG/Project_RL.git

https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016

PPT 보고서 : 강화학습 Project_A70027강보현.pptx[https://github.com/bhonSG/Project_RL/issues/1#issue-3703480016]

업로드 파일
 1) user_seg.csv
 2) ad_watch_hist.csv

소스 파일 : RLProject_A70027.py

실행방법
 1) git clone, venv 활성화(.\venv\Scripts\activate), 미포함 패키지 import
 2) 스크립트 실행: python RLProject_A70027.py
 + Python 버전: Python 3.12.2
 + 설치 패키지/버전
    keras==3.12.0
    matplotlib==3.10.7
    numpy==2.3.5
    pandas==2.3.3
    tensorflow==2.20.0


수행내역

================================================== 원본 데이터 확인
user_seg_df 컬럼: ['user_id', 'seg_id'] user_seg_df 샘플: user_id seg_id 0 1 C0 1 2 C0 2 3 G0 3 4 _ 4 5 C0

ad_watch_hist_df 컬럼: ['user_id', 'ad_id', 'ad_brand', 'full_watch_cnt'] ad_watch_hist_df 샘플: user_id ad_id ad_brand full_watch_cnt 0 1 1 23 1 1 2 1 23 1 2 3 1 23 1 3 4 1 23 1 4 5 1 996 1

병합된 데이터 shape: (11582, 5) 병합된 데이터 컬럼: ['user_id', 'ad_id', 'ad_brand', 'full_watch_cnt', 'seg_id'] user_id ad_id ad_brand full_watch_cnt seg_id 0 1 1 23 1 C0 1 2 1 23 1 C0 2 3 1 23 1 G0 3 4 1 23 1 _ 4 4 1 23 1 A0 5 5 1 996 1 C0 6 6 1 70 1 C0 7 7 1 999 1 C0 8 8 1 24 1 _ 9 9 1 37 1 _

================================================== State & Action 정의
세그먼트 수: 12 광고 브랜드/카테고리 수: 101

Segment 매핑: {'A0': 0, 'B0': 1, 'C0': 2, 'D0': 3, 'E0': 4, 'F0': 5, 'G0': 6, 'H0': 7, 'I0': 8, 'J0': 9, 'K0': 10, '_': 11}
Brand/Category 매핑: {np.int64(0): 0, np.int64(1): 1, np.int64(3): 2, np.int64(15): 3, np.int64(16): 4, np.int64(18): 5, np.int64(19): 6, np.int64(21): 7, np.int64(23): 8, np.int64(24): 9, np.int64(25): 10, np.int64(27): 11, np.int64(31): 12, np.int64(33): 13, np.int64(35): 14, np.int64(37): 15, np.int64(39): 16, np.int64(40): 17, np.int64(41): 18, np.int64(42): 19, np.int64(43): 20, np.int64(45): 21, np.int64(46): 22, np.int64(47): 23, np.int64(48): 24, np.int64(51): 25, np.int64(52): 26, np.int64(53): 27, np.int64(54): 28, np.int64(55): 29, np.int64(56): 30, np.int64(57): 31, np.int64(58): 32, np.int64(59): 33, np.int64(60): 34, np.int64(61): 35, np.int64(62): 36, np.int64(63): 37, np.int64(66): 38, np.int64(67): 39, np.int64(68): 40, np.int64(69): 41, np.int64(70): 42, np.int64(72): 43, np.int64(73): 44, np.int64(75): 45, np.int64(76): 46, np.int64(78): 47, np.int64(80): 48, np.int64(81): 49, np.int64(82): 50, np.int64(83): 51, np.int64(84): 52, np.int64(85): 53, np.int64(86): 54, np.int64(87): 55, np.int64(88): 56, np.int64(89): 57, np.int64(90): 58, np.int64(91): 59, np.int64(98): 60, np.int64(102): 61, np.int64(103): 62, np.int64(104): 63, np.int64(106): 64, np.int64(108): 65, np.int64(110): 66, np.int64(112): 67, np.int64(114): 68, np.int64(118): 69, np.int64(120): 70, np.int64(121): 71, np.int64(122): 72, np.int64(123): 73, np.int64(126): 74, np.int64(127): 75, np.int64(128): 76, np.int64(130): 77, np.int64(136): 78, np.int64(137): 79, np.int64(140): 80, np.int64(145): 81, np.int64(146): 82, np.int64(159): 83, np.int64(164): 84, np.int64(176): 85, np.int64(180): 86, np.int64(181): 87, np.int64(183): 88, np.int64(184): 89, np.int64(185): 90, np.int64(186): 91, np.int64(276): 92, np.int64(984): 93, np.int64(990): 94, np.int64(991): 95, np.int64(994): 96, np.int64(995): 97, np.int64(996): 98, np.int64(998): 99, np.int64(999): 100}

================================================== Reward 정의 (full_watch_cnt 기반)
완전시청(1) 비율: 74.27% 불완전시청(0) 비율: 25.73%

Reward 분포: reward 0 2980 1 8602 Name: count, dtype: int64

================================================== 변환된 데이터 정보
상태 수 (Segment): 12 액션 수 (Ad IDs): 230 총 샘플 수: 11582

변환된 데이터: user_id seg_id ad_id ad_brand full_watch_cnt state action reward 0 1 C0 1 23 1 2 0 1 1 2 C0 1 23 1 2 0 1 2 3 G0 1 23 1 6 0 1 3 4 _ 1 23 1 11 0 1 4 4 A0 1 23 1 0 0 1 5 5 C0 1 996 1 2 0 1 6 6 C0 1 70 1 2 0 1 7 7 C0 1 999 1 2 0 1 8 8 _ 1 24 1 11 0 1 9 9 _ 1 37 1 11 0 1 10 10 D0 1 0 1 3 0 1 11 8 _ 1 56 1 11 0 1 12 11 _ 1 56 1 11 0 1 13 12 _ 1 41 1 11 0 1 14 13 _ 1 62 1 11 0 1

================================================== Q-Learning 훈련 시작 (Exponential Epsilon Decay)
Q-Learning Episode 10/400, Reward: 151, Avg: 0.7550, Epsilon: 0.8171 Q-Learning Episode 20/400, Reward: 152, Avg: 0.7600, Epsilon: 0.6676 Q-Learning Episode 30/400, Reward: 152, Avg: 0.7600, Epsilon: 0.5455 Q-Learning Episode 40/400, Reward: 137, Avg: 0.6850, Epsilon: 0.4457 Q-Learning Episode 50/400, Reward: 143, Avg: 0.7150, Epsilon: 0.3642 Q-Learning Episode 60/400, Reward: 144, Avg: 0.7200, Epsilon: 0.2976 Q-Learning Episode 70/400, Reward: 143, Avg: 0.7150, Epsilon: 0.2431 Q-Learning Episode 80/400, Reward: 154, Avg: 0.7700, Epsilon: 0.1986 Q-Learning Episode 90/400, Reward: 155, Avg: 0.7750, Epsilon: 0.1623 Q-Learning Episode 100/400, Reward: 145, Avg: 0.7250, Epsilon: 0.1326 Q-Learning Episode 110/400, Reward: 149, Avg: 0.7450, Epsilon: 0.1084 Q-Learning Episode 120/400, Reward: 152, Avg: 0.7600, Epsilon: 0.0885 Q-Learning Episode 130/400, Reward: 152, Avg: 0.7600, Epsilon: 0.0723 Q-Learning Episode 140/400, Reward: 155, Avg: 0.7750, Epsilon: 0.0591 Q-Learning Episode 150/400, Reward: 150, Avg: 0.7500, Epsilon: 0.0500 Q-Learning Episode 160/400, Reward: 147, Avg: 0.7350, Epsilon: 0.0500 Q-Learning Episode 170/400, Reward: 148, Avg: 0.7400, Epsilon: 0.0500 Q-Learning Episode 180/400, Reward: 150, Avg: 0.7500, Epsilon: 0.0500 Q-Learning Episode 190/400, Reward: 150, Avg: 0.7500, Epsilon: 0.0500 Q-Learning Episode 200/400, Reward: 151, Avg: 0.7550, Epsilon: 0.0500 Q-Learning Episode 210/400, Reward: 138, Avg: 0.6900, Epsilon: 0.0500 Q-Learning Episode 220/400, Reward: 160, Avg: 0.8000, Epsilon: 0.0500 Q-Learning Episode 230/400, Reward: 146, Avg: 0.7300, Epsilon: 0.0500 Q-Learning Episode 240/400, Reward: 153, Avg: 0.7650, Epsilon: 0.0500 Q-Learning Episode 250/400, Reward: 161, Avg: 0.8050, Epsilon: 0.0500 Q-Learning Episode 260/400, Reward: 146, Avg: 0.7300, Epsilon: 0.0500 Q-Learning Episode 270/400, Reward: 142, Avg: 0.7100, Epsilon: 0.0500 Q-Learning Episode 280/400, Reward: 144, Avg: 0.7200, Epsilon: 0.0500 Q-Learning Episode 290/400, Reward: 140, Avg: 0.7000, Epsilon: 0.0500 Q-Learning Episode 300/400, Reward: 143, Avg: 0.7150, Epsilon: 0.0500 Q-Learning Episode 310/400, Reward: 152, Avg: 0.7600, Epsilon: 0.0500 Q-Learning Episode 320/400, Reward: 151, Avg: 0.7550, Epsilon: 0.0500 Q-Learning Episode 330/400, Reward: 147, Avg: 0.7350, Epsilon: 0.0500 Q-Learning Episode 340/400, Reward: 153, Avg: 0.7650, Epsilon: 0.0500 Q-Learning Episode 350/400, Reward: 159, Avg: 0.7950, Epsilon: 0.0500 Q-Learning Episode 360/400, Reward: 153, Avg: 0.7650, Epsilon: 0.0500 Q-Learning Episode 370/400, Reward: 148, Avg: 0.7400, Epsilon: 0.0500 Q-Learning Episode 380/400, Reward: 150, Avg: 0.7500, Epsilon: 0.0500 Q-Learning Episode 390/400, Reward: 145, Avg: 0.7250, Epsilon: 0.0500 Q-Learning Episode 400/400, Reward: 151, Avg: 0.7550, Epsilon: 0.0500

Q-Learning 훈련 완료 최종 에피소드 보상: 151 최종 평균 보상: 0.7550

================================================== DQN 훈련 시작
하이퍼파라미터:

Learning Rate: 0.0005
Batch Size: 16
Target Update Interval: 60 episodes
Epsilon Decay: 0.98 (exponential)
Dense Layer: 64×64
Episodes: 100 ================================================== DQN Episode 10/100, Reward: 72, Avg: 0.7200, Epsilon: 0.8171, Memory: 1000 DQN Episode 20/100, Reward: 69, Avg: 0.6900, Epsilon: 0.6676, Memory: 2000 DQN Episode 30/100, Reward: 70, Avg: 0.7000, Epsilon: 0.5455, Memory: 3000 DQN Episode 40/100, Reward: 76, Avg: 0.7600, Epsilon: 0.4457, Memory: 4000 DQN Episode 50/100, Reward: 74, Avg: 0.7400, Epsilon: 0.3642, Memory: 5000 DQN Episode 60/100, Reward: 66, Avg: 0.6600, Epsilon: 0.2976, Memory: 5000 DQN Episode 70/100, Reward: 78, Avg: 0.7800, Epsilon: 0.2431, Memory: 5000 DQN Episode 80/100, Reward: 62, Avg: 0.6200, Epsilon: 0.1986, Memory: 5000 DQN Episode 90/100, Reward: 75, Avg: 0.7500, Epsilon: 0.1623, Memory: 5000 DQN Episode 100/100, Reward: 63, Avg: 0.6300, Epsilon: 0.1326, Memory: 5000
DQN 훈련 완료 최종 에피소드 보상: 63 최종 평균 보상: 0.6300

================================================== 정책 평가
[Q-Learning] 평가 시작...

[QL] 평가 시작... 총 11582 샘플 처리 중 [QL] 1000 / 11582 처리 완료... [QL] 2000 / 11582 처리 완료... [QL] 3000 / 11582 처리 완료... [QL] 4000 / 11582 처리 완료... [QL] 5000 / 11582 처리 완료... [QL] 6000 / 11582 처리 완료... [QL] 7000 / 11582 처리 완료... [QL] 8000 / 11582 처리 완료... [QL] 9000 / 11582 처리 완료... [QL] 10000 / 11582 처리 완료... [QL] 11000 / 11582 처리 완료... [QL] 평가 완료! 샘플 처리: 11582 액션 매칭: 1150 / 11582 (9.93%) 누적 보상: 1150 완전시청 수: 1150

Q-Learning 평가 결과: 평균 Reward: 1.0000 완전시청 달성률: 100.00%

[DQN] 평가 시작...

[DQN] 평가 시작... 총 11582 샘플 처리 중 [DQN] 1000 / 11582 처리 완료... [DQN] 2000 / 11582 처리 완료... [DQN] 3000 / 11582 처리 완료... [DQN] 4000 / 11582 처리 완료... [DQN] 5000 / 11582 처리 완료... [DQN] 6000 / 11582 처리 완료... [DQN] 7000 / 11582 처리 완료... [DQN] 8000 / 11582 처리 완료... [DQN] 9000 / 11582 처리 완료... [DQN] 10000 / 11582 처리 완료... [DQN] 11000 / 11582 처리 완료... [DQN] 평가 완료! 샘플 처리: 11582 액션 매칭: 216 / 11582 (1.86%) 누적 보상: 216 완전시청 수: 216

DQN (Optimized) 평가 결과: 평균 Reward: 1.0000 완전시청 달성률: 100.00%

================================================== 각 고객 세그먼트별 최적 광고 추천 (Q-Learning)
Segment 'A0': 추천 광고 ID 2 (Q-value: 1.0000) Segment 'B0': 추천 광고 ID 4 (Q-value: 1.0000) Segment 'C0': 추천 광고 ID 1 (Q-value: 1.0000) Segment 'D0': 추천 광고 ID 2 (Q-value: 1.0000) Segment 'E0': 추천 광고 ID 58 (Q-value: 1.0000) Segment 'F0': 추천 광고 ID 4 (Q-value: 1.0000) Segment 'G0': 추천 광고 ID 4 (Q-value: 0.9993) Segment 'H0': 추천 광고 ID 1 (Q-value: 0.9927) Segment 'I0': 추천 광고 ID 3 (Q-value: 1.0000) Segment 'J0': 추천 광고 ID 4 (Q-value: 1.0000) Segment 'K0': 추천 광고 ID 4 (Q-value: 1.0000) Segment '_': 추천 광고 ID 1 (Q-value: 0.9997)

================================================== 각 고객 세그먼트별 최적 광고 추천 (DQN - Optimized)
Segment 'A0': 추천 광고 ID 56 (Q-value: 2.3082) Segment 'B0': 추천 광고 ID 56 (Q-value: 2.3431) Segment 'C0': 추천 광고 ID 1 (Q-value: 2.0403) Segment 'D0': 추천 광고 ID 129 (Q-value: 2.2901) Segment 'E0': 추천 광고 ID 39 (Q-value: 2.4187) Segment 'F0': 추천 광고 ID 97 (Q-value: 2.3492) Segment 'G0': 추천 광고 ID 56 (Q-value: 2.2871) Segment 'H0': 추천 광고 ID 1 (Q-value: 2.0919) Segment 'I0': 추천 광고 ID 1 (Q-value: 2.4762) Segment 'J0': 추천 광고 ID 125 (Q-value: 2.3948) Segment 'K0': 추천 광고 ID 56 (Q-value: 2.2614) Segment '_': 추천 광고 ID 56 (Q-value: 2.0794)

================================================== 학습 완료!
