import pygame
import time
from stable_baselines3 import PPO
from environment.angry_birds_environment import AngryBirdsEnv

# Paths to images
BACKGROUND_IMAGE_PATH = r"C:\Users\krish\OneDrive\Desktop\Credenz\Credenz-25-Xodia\ui\images\bg.jpg"
SLINGSHOT_IMAGE_PATH = r"C:\Users\krish\OneDrive\Desktop\Credenz\Credenz-25-Xodia\ui\images\sling.png"
LOGO_IMAGE_PATH = r"C:\Users\krish\OneDrive\Desktop\Credenz\Credenz-25-Xodia\ui\images\credenz-logo.png"

def test_model(model_path="angry_birds_ppo_y0.zip"):
    """Load a trained PPO model and test it in the Angry Birds environment with full UI."""
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 450))  # Game window size
    pygame.display.set_caption("Angry Birds PPO Agent")

    # Load images with error handling
    try:
        background = pygame.image.load(BACKGROUND_IMAGE_PATH)
        slingshot = pygame.image.load(SLINGSHOT_IMAGE_PATH)
        slingshot = pygame.transform.scale(slingshot, (80, 150))  # Resize slingshot
        logo = pygame.image.load(LOGO_IMAGE_PATH)
        logo = pygame.transform.scale(logo, (120, 50))  # Resize logo
    except pygame.error as e:
        print(f"Error loading image: {e}")
        print(f"Please verify these files exist:")
        print(f"- {BACKGROUND_IMAGE_PATH}")
        print(f"- {SLINGSHOT_IMAGE_PATH}")
        print(f"- {LOGO_IMAGE_PATH}")
        pygame.quit()
        return

    # Create environment
    env = AngryBirdsEnv()
    model = PPO.load(model_path)  # Load trained model
    i = 0
    counter = 0
    while(i<100):
        i=i+1
        obs = env.reset()[0]
        done = False
        score = 0  # Start with a score of 0
        
        # Main testing loop
        while not done:
            screen.blit(background, (0, 0))  # Draw background
            screen.blit(slingshot, (50, 200))  # Draw slingshot
            screen.blit(logo, (650, 10))  # Draw logo
            
            env.bird.draw(screen)  # Draw bird
            env.pig.draw(screen)  # Draw pig

            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            # If bird hits the pig, set score to 200
            if reward >= 100:  # We assume hitting the pig gives 100 reward
                score = 200
                counter += 1
            elif done:  # If episode ends and no hit happened, keep score at 0
                score = 0

            # Display score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {score}", True, (0, 0, 0))
            screen.blit(score_text, (20, 20))

            pygame.display.flip()  # Update screen
            time.sleep(0.05)  # Small delay for smooth animation

        # time.sleep(0.5)
        print(f"Final Score: {score}")
    print("Hit rate: ", counter)
    pygame.quit()  # Close Pygame after testing


if __name__ == "__main__":
    test_model()  # Run test when script is executed
