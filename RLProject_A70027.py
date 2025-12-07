import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from collections import defaultdict
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
import matplotlib.pyplot as plt

# 데이터 로드
user_seg_df = pd.read_csv('user_seg.csv')
ad_watch_hist_df = pd.read_csv('ad_watch_hist.csv')

print("=" * 50)
print("원본 데이터 확인")
print("=" * 50)
print("\nuser_seg_df 컬럼:", user_seg_df.columns.tolist())
print("user_seg_df 샘플:")
print(user_seg_df.head())

print("\nad_watch_hist_df 컬럼:", ad_watch_hist_df.columns.tolist())
print("ad_watch_hist_df 샘플:")
print(ad_watch_hist_df.head())

# 데이터 병합
merged_df = ad_watch_hist_df.merge(user_seg_df, on='user_id', how='left')
merged_df = merged_df.dropna()

print("\n병합된 데이터 shape:", merged_df.shape)
print("병합된 데이터 컬럼:", merged_df.columns.tolist())
print(merged_df.head(10))

# ============= State 구성: seg_id 기반 =============
# seg_id와 ad_brand 조합으로 상태 생성
seg_ids = sorted(merged_df['seg_id'].unique())
ad_brands = sorted(merged_df['ad_brand'].unique())

seg_mapping = {seg: idx for idx, seg in enumerate(seg_ids)}
brand_mapping = {brand: idx for idx, brand in enumerate(ad_brands)}

num_seg_states = len(seg_mapping)
num_brand_actions = len(brand_mapping)

print("\n" + "=" * 50)
print("State & Action 정의")
print("=" * 50)
print(f"세그먼트 수: {num_seg_states}")
print(f"광고 브랜드/카테고리 수: {num_brand_actions}")

print(f"\nSegment 매핑: {seg_mapping}")
print(f"Brand/Category 매핑: {brand_mapping}")

# ============= Reward 정의: full_watch_cnt 기반 =============
# full_watch_cnt: 1(완전시청), 0(불완전시청)
merged_df['reward'] = merged_df['full_watch_cnt'].astype(int)

print(f"\n" + "=" * 50)
print("Reward 정의 (full_watch_cnt 기반)")
print("=" * 50)
print(f"완전시청(1) 비율: {(merged_df['reward'] == 1).sum() / len(merged_df):.2%}")
print(f"불완전시청(0) 비율: {(merged_df['reward'] == 0).sum() / len(merged_df):.2%}")

print(f"\nReward 분포:")
print(merged_df['reward'].value_counts().sort_index())

# ============= State & Action 인코딩 =============
# State: 고객 세그먼트 (seg_id)
merged_df['state'] = merged_df['seg_id'].map(seg_mapping)

# Action: 광고 선택 (ad_id)
ad_id_mapping = {ad_id: idx for idx, ad_id in enumerate(sorted(merged_df['ad_id'].unique()))}
merged_df['action'] = merged_df['ad_id'].map(ad_id_mapping)

num_states = num_seg_states
num_actions = len(ad_id_mapping)

print("\n" + "=" * 50)
print("변환된 데이터 정보")
print("=" * 50)
print(f"상태 수 (Segment): {num_states}")
print(f"액션 수 (Ad IDs): {num_actions}")
print(f"총 샘플 수: {len(merged_df)}")

print("\n변환된 데이터:")
print(merged_df[['user_id', 'seg_id', 'ad_id', 'ad_brand', 'full_watch_cnt',
                  'state', 'action', 'reward']].head(15))

# ============= Q-Learning 구현 =============
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.05, discount_factor=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.episode_rewards = []
        self.episode_avg_rewards = []
        
    def select_action(self, state, epsilon=0.1):
        """ε-greedy 정책으로 액션 선택"""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-value 업데이트"""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])
    
    def train(self, data, episodes=400, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.98):
        """Q-Learning 훈련 (Exponential Epsilon Decay)"""
        epsilon = epsilon_start
        
        for episode in range(episodes):
            episode_reward = 0
            episode_count = 0
            
            # 각 에피소드마다 샘플링
            sampled_data = data.sample(n=min(len(data), 200), replace=True).reset_index(drop=True)
            
            for idx, row in sampled_data.iterrows():
                state = int(row['state'])
                action = int(row['action'])
                reward = int(row['reward'])
                next_state = np.random.randint(self.num_states)
                done = True
                
                self.update(state, action, reward, next_state, done)
                episode_reward += reward
                episode_count += 1
            
            avg_reward = episode_reward / episode_count if episode_count > 0 else 0
            self.episode_rewards.append(episode_reward)
            self.episode_avg_rewards.append(avg_reward)
            
            # Exponential decay
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            if (episode + 1) % 10 == 0:
                print(f"Q-Learning Episode {episode + 1}/{episodes}, Reward: {episode_reward}, "
                      f"Avg: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")
        
        return self.episode_rewards, self.episode_avg_rewards

# Q-Learning 훈련
print("\n" + "=" * 50)
print("Q-Learning 훈련 시작 (Exponential Epsilon Decay)")
print("=" * 50)

q_agent = QLearningAgent(num_states, num_actions)
q_rewards, q_avg_rewards = q_agent.train(merged_df, episodes=400, 
                                         epsilon_start=1.0, 
                                         epsilon_end=0.05, 
                                         epsilon_decay=0.98)

print("\nQ-Learning 훈련 완료")
print(f"최종 에피소드 보상: {q_rewards[-1]}")
print(f"최종 평균 보상: {q_avg_rewards[-1]:.4f}")

# ============= DQN 구현 (최적화 버전) =============
class DQNAgent:
    def __init__(self, num_states, num_actions, 
                 learning_rate=0.0005, 
                 discount_factor=0.99,
                 max_memory=5000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory = []
        self.max_memory = max_memory
        self.episode_rewards = []
        self.episode_avg_rewards = []
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
    
    def build_model(self):
        """DQN 신경망 구축 (단순화된 64×64 구조)"""
        model = keras.Sequential([
            keras.layers.Input(shape=(1,), dtype='int32'),
            keras.layers.Embedding(input_dim=self.num_states, output_dim=16),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.num_actions)
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='mse')
        return model
    
    def update_target_model(self):
        """타겟 모델 업데이트"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((int(state), int(action), float(reward), int(next_state), bool(done)))
    
    def select_action(self, state, epsilon=0.1):
        """ε-greedy 정책으로 액션 선택"""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.model.predict(np.array([[state]], dtype='int32'), verbose=0)
            return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """경험 재생을 통한 훈련"""
        if len(self.memory) == 0:
            return
        
        batch_size = min(batch_size, len(self.memory))
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0] for i in batch_indices]).reshape(-1, 1).astype('int32')
        actions = np.array([self.memory[i][1] for i in batch_indices]).astype('int32')
        rewards = np.array([self.memory[i][2] for i in batch_indices]).astype('float32')
        next_states = np.array([self.memory[i][3] for i in batch_indices]).reshape(-1, 1).astype('int32')
        
        target = self.model.predict(states, verbose=0).astype('float32')
        target_next = self.target_model.predict(next_states, verbose=0).astype('float32')
        
        for i in range(batch_size):
            target[i, actions[i]] = rewards[i] + self.discount_factor * np.max(target_next[i])
        
        self.model.fit(states, target, epochs=1, verbose=0)
    
    def train(self, data, episodes=100, batch_size=16, 
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.98,
              target_update_interval=60):
        """DQN 훈련"""
        epsilon = epsilon_start
        
        for episode in range(episodes):
            episode_reward = 0
            episode_count = 0
            
            # 한 에피소드에서 사용하는 샘플 수 축소 (속도 개선)
            sampled_data = data.sample(n=min(len(data), 100), replace=True).reset_index(drop=True)
            
            for idx, row in sampled_data.iterrows():
                state = int(row['state'])
                action = int(row['action'])
                reward = int(row['reward'])
                next_state = np.random.randint(self.num_states)
                done = True
                
                self.remember(state, action, reward, next_state, done)
                
                # 메모리가 충분하면 배치로 훈련
                if len(self.memory) >= batch_size:
                    self.replay(batch_size)
                
                episode_reward += reward
                episode_count += 1
            
            avg_reward = episode_reward / episode_count if episode_count > 0 else 0
            self.episode_rewards.append(episode_reward)
            self.episode_avg_rewards.append(avg_reward)
            
            # Exponential epsilon decay
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # target network 업데이트 빈도 완화 (더 적게 업데이트)
            if (episode + 1) % target_update_interval == 0:
                self.update_target_model()
            
            if (episode + 1) % 10 == 0:
                print(f"DQN Episode {episode + 1}/{episodes}, Reward: {episode_reward}, "
                      f"Avg: {avg_reward:.4f}, Epsilon: {epsilon:.4f}, Memory: {len(self.memory)}")
        
        return self.episode_rewards, self.episode_avg_rewards

# DQN 훈련
print("\n" + "=" * 50)
print("DQN 훈련 시작")
print("=" * 50)
print("하이퍼파라미터:")
print("  - Learning Rate: 0.0005")
print("  - Batch Size: 16")
print("  - Target Update Interval: 60 episodes")
print("  - Epsilon Decay: 0.98 (exponential)")
print("  - Dense Layer: 64×64")
print("  - Episodes: 100")
print("=" * 50)

dqn_agent = DQNAgent(num_states, num_actions)
dqn_rewards, dqn_avg_rewards = dqn_agent.train(merged_df, episodes=100, 
                                               batch_size=16,
                                               epsilon_start=1.0,
                                               epsilon_end=0.05,
                                               epsilon_decay=0.98,
                                               target_update_interval=60)

print("\nDQN 훈련 완료")
print(f"최종 에피소드 보상: {dqn_rewards[-1]}")
print(f"최종 평균 보상: {dqn_avg_rewards[-1]:.4f}")

# ============= 정책 평가 =============
print("\n" + "=" * 50)
print("정책 평가")
print("=" * 50)

def evaluate_policy(agent, data, agent_type='QL'):
    """정책 평가"""
    total_reward = 0
    total_count = 0
    full_watch_count = 0
    matched_count = 0
    
    print(f"\n[{agent_type}] 평가 시작... 총 {len(data)} 샘플 처리 중")
    
    for idx, row in data.iterrows():
        state = int(row['state'])
        true_action = int(row['action'])
        true_reward = int(row['reward'])
        
        try:
            if agent_type == 'QL':
                predicted_action = np.argmax(agent.q_table[state])
            else:  # DQN
                q_values = agent.model.predict(np.array([[state]], dtype='int32'), verbose=0)
                predicted_action = int(np.argmax(q_values[0]))
            
            if predicted_action == true_action:
                matched_count += 1
                total_reward += true_reward
                if true_reward == 1:
                    full_watch_count += 1
        
        except Exception as e:
            if idx < 3:  # 처음 3개 오류만 출력
                print(f"[ERROR] idx={idx}, state={state}: {e}")
            continue
        
        total_count += 1
        
        # 진행률 표시 (매 1000개마다)
        if (idx + 1) % 1000 == 0:
            print(f"  [{agent_type}] {idx + 1} / {len(data)} 처리 완료...")
    
    print(f"[{agent_type}] 평가 완료")
    
    avg_reward = total_reward / matched_count if matched_count > 0 else 0
    full_watch_rate = full_watch_count / matched_count * 100 if matched_count > 0 else 0
    action_match_rate = matched_count / total_count * 100 if total_count > 0 else 0
    
    print(f"  샘플 처리: {total_count}")
    print(f"  액션 매칭: {matched_count} / {total_count} ({action_match_rate:.2f}%)")
    print(f"  누적 보상: {total_reward}")
    print(f"  완전시청 수: {full_watch_count}")
    
    return avg_reward, full_watch_rate

print("\n[Q-Learning] 평가 시작...")
q_avg_reward, q_full_watch = evaluate_policy(q_agent, merged_df, 'QL')

print(f"\nQ-Learning 평가 결과:")
print(f"  평균 Reward: {q_avg_reward:.4f}")
print(f"  완전시청 달성률: {q_full_watch:.2f}%")

print("\n[DQN] 평가 시작...")
dqn_avg_reward, dqn_full_watch = evaluate_policy(dqn_agent, merged_df, 'DQN')

print(f"\nDQN (Optimized) 평가 결과:")
print(f"  평균 Reward: {dqn_avg_reward:.4f}")
print(f"  완전시청 달성률: {dqn_full_watch:.2f}%")

# ============= 최적 정책 추출 =============
print("\n" + "=" * 50)
print("각 고객 세그먼트별 최적 광고 추천 (Q-Learning)")
print("=" * 50)

inverse_seg_mapping = {v: k for k, v in seg_mapping.items()}
inverse_ad_mapping = {v: k for k, v in ad_id_mapping.items()}

for state in range(num_states):
    if len(q_agent.q_table[state]) > 0:
        best_action = np.argmax(q_agent.q_table[state])
        best_q_value = np.max(q_agent.q_table[state])
        seg_name = inverse_seg_mapping[state]
        ad_id = inverse_ad_mapping.get(best_action, best_action)
        print(f"Segment '{seg_name}': 추천 광고 ID {ad_id} (Q-value: {best_q_value:.4f})")

print("\n" + "=" * 50)
print("각 고객 세그먼트별 최적 광고 추천 (DQN - Optimized)")
print("=" * 50)

for state in range(num_states):
    q_values = dqn_agent.model.predict(np.array([[state]], dtype='int32'), verbose=0)
    best_action = np.argmax(q_values[0])
    best_q_value = np.max(q_values[0])
    seg_name = inverse_seg_mapping[state]
    ad_id = inverse_ad_mapping.get(best_action, best_action)
    print(f"Segment '{seg_name}': 추천 광고 ID {ad_id} (Q-value: {best_q_value:.4f})")

# ============= 결과 시각화 =============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Q-Learning 총 보상
axes[0, 0].plot(q_rewards, label='Q-Learning', marker='o', markersize=3, linewidth=2)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].set_title('Q-Learning Training Progress (Total Reward)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Q-Learning 평균 보상
axes[0, 1].plot(q_avg_rewards, label='Q-Learning Avg', marker='o', markersize=3, 
                linewidth=2, color='green')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Average Reward')
axes[0, 1].set_title('Q-Learning Training Progress (Average Reward)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# DQN 총 보상
axes[1, 0].plot(dqn_rewards, label='DQN (Optimized)', marker='s', markersize=3, 
                linewidth=2, color='orange')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Total Reward')
axes[1, 0].set_title('DQN Training Progress (Total Reward)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# DQN 평균 보상
axes[1, 1].plot(dqn_avg_rewards, label='DQN (Optimized) Avg', marker='s', markersize=3, 
                linewidth=2, color='red')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Average Reward')
axes[1, 1].set_title('DQN Training Progress (Average Reward)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_training_results.png', dpi=100, bbox_inches='tight')
plt.show()

# Q-Learning vs DQN 비교
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(q_avg_rewards, label='Q-Learning', linewidth=2.5, marker='o', markersize=2, alpha=0.8)
ax.plot(dqn_avg_rewards, label='DQN (Optimized)', linewidth=2.5, marker='s', markersize=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Average Reward', fontsize=12)
ax.set_title('Q-Learning vs DQN (Optimized): Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('rl_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("학습 완료")
print("=" * 50)