from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.angry_birds_environment import AngryBirdsEnv
import os

def train_model():
    env = make_vec_env(AngryBirdsEnv, n_envs=4)  # Parallel training with 4 environments

    model_path = "angry_birds_ppo.zip"

    if os.path.exists(model_path):  
        print("✅ Existing model found! Continuing training...")
        model = PPO.load(model_path, env=env)  # Load existing model
    else:
        print("🚀 No existing model found. Training from scratch...")
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=0.00015,   # 🔹 Lower LR for smoother, more controlled learning
            gamma=0.985,             # 🔹 Slightly shorter-term focus (better launch adjustments)
            batch_size=256,          # 🔹 Bigger batch = more stable learning
            n_steps=4096,            # 🔹 Longer trajectory buffer (learns better movement patterns)
            n_epochs=7,              # 🔹 Enough updates per batch to learn adjustments
            clip_range=0.15,         # 🔹 More controlled policy updates
            ent_coef=0.0005,         # 🔹 Slightly less exploration (precision)
            vf_coef=0.6,             # 🔹 Higher value function importance (better stability)
            max_grad_norm=0.3        # 🔹 Stricter gradient clipping (prevents unstable updates)
        )

    model.learn(total_timesteps=500000)  # Train for 2.5M timesteps

    model.save("angry_birds_ppo")  # Save updated model

if __name__ == "__main__":
    train_model()
