# Credenz'25 Xodia - Angry Birds RL Competition

Welcome to the **Angry Birds RL Competition!** 🎯🐦  
This competition challenges participants to develop **Reinforcement Learning (RL) agents** that can efficiently launch birds to hit pigs using a physics-based environment.

## 📌 **Game Overview**
The environment simulates an **Angry Birds**-style game where:
- A bird is launched from a **slingshot** towards a **target pig**.
- The bird follows a **trajectory influenced by gravity**.
- Players control the **launch power and angle** to maximize accuracy.

---

## 🏗 **Observation Space**
The environment provides an **observation space** represented as a 8-dimensional vector:

| **Feature**         | **Description**                                   | **Range** |
|---------------------|---------------------------------------------------|-----------|
| `bird_x`           | Bird's x-coordinate                               | `[0, 800]` |
| `bird_y`           | Bird's y-coordinate                               | `[0, 450]` |
| `velocity_x`       | Bird's x-axis velocity                            | `[-30, 30]` |
| `velocity_y`       | Bird's y-axis velocity                            | `[-30, 30]` |
| `pig_x`            | Pig's x-coordinate                                | `[0, 800]` |
| `pig_y`            | Pig's y-coordinate                                | `[0, 450]` |
| `max_height`       | Maximum height reached by the bird                | `[0, 450]` |
| `launch_angle`     | Bird's launch angle (radians)                     | `[-π, π]` |

---

## 🎮 **Action Space**
Participants control the bird's **launch power** along the x and y axes:

| **Action**  | **Description**           | **Range** |
|------------|--------------------------|-----------|
| `power_x`  | Launch power in x-direction | `[1, 15]` |
| `power_y`  | Launch power in y-direction | `[1, 15]` |

- A **higher power_x** pushes the bird farther horizontally.
- A **higher power_y** results in a higher arc.

---

## 🏆 **Scoring & Rewards**
Your RL agent earns **rewards** based on performance:

- ✅ **+200** for hitting the pig.
- 🎯 **+0.5 per unit** decrease in distance to the pig.
- 🏹 **Bonus reward** for achieving interesting trajectories.
- ❌ **-5** for going out of bounds.
- ⏳ **Episode ends** when:
  - The bird hits the pig.
  - The bird goes out of bounds.
  - The maximum number of steps (200) is reached.

---

## 🛠 Implementation Tasks

To successfully participate, complete the following key components:

### 1️⃣ Implement the Training Loop (`train.py`)
- Load the **Angry Birds RL environment**.
- Initialize the **RL agent**.
- Implement the **training loop** where the agent:
  - Selects actions based on its policy.
  - Interacts with the environment.
  - Collects rewards and updates its model.
- Save the trained model for evaluation.

### 2️⃣ Complete the Reward Function (`AngryBirdsEnv`)
- Ensure rewards follow the scoring system outlined above.
- Implement logic for:
  - **Positive rewards** for hitting the pig.
  - **Penalty for going out of bounds**.
  - **Reward shaping** to encourage better trajectories.

--- 

Good luck to all the participants!


