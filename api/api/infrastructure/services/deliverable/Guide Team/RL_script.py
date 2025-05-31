import os
import json
import random
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from torch.distributions import Categorical
from dotenv import load_dotenv

# Load variables from helper files
from Bins import CONCEPT_BINS, LEARNING_STYLE_BINS
from Block_options import BLOCK_OPTIONS
from Prompt_template import PROMPT_TEMPLATE
from Rubric import RUBRIC_TEXT

# Load OpenAPI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Define constants and configurations 
DEVICE = "cpu" 
LR = 3e-4  
HIDDEN_SIZE = 64  
ENTROPY_LAMBDA = 0.05  # Improved for more exploration
BASELINE_BETA = 0.9 
N_EPISODES = 50  
TEMPERATURE_LLM = 0.3  # Improved for more diverse responses
MODEL_NAME_LLM = "gpt-4o-mini"  
EVAL_RETRIES = 3  
SAVE_INTERVAL = 10  
LOG_INTERVAL = 5 

# Improved Exploration Parameters
EPSILON_START = 0.3  # Start with 30% random actions
EPSILON_END = 0.05   # End with 5% random actions
EPSILON_DECAY = 0.995  # How fast to decay exploration

# Improved Reward scaling parameters
REWARD_SCALE_FACTOR = 0.8  # Scale down rewards to avoid saturation
REWARD_NOISE_STD = 2.0     # Add noise to prevent identical rewards
USE_COMPARATIVE_GRADING = True  # Compare against baseline responses

# Parse bins from the imported strings
def parse_bins(bin_string: str) -> List[str]:
    lines = bin_string.strip().split('\n')
    bins = []
    for line in lines:
        line = line.strip()
        if line:  # Ensure the line is not empty
            # Remove any leading bullets or dashes
            clean_line = line.lstrip('- ').strip()
            bins.append(clean_line)
    return bins

# Parse the actual bins
CONCEPT_BIN_LIST = parse_bins(CONCEPT_BINS)
LEARNING_STYLE_BIN_LIST = parse_bins(LEARNING_STYLE_BINS)

# Create mappings
BIN2IDX_CONCEPT = {b: i for i, b in enumerate(CONCEPT_BIN_LIST)}
BIN2IDX_STYLE = {b: i for i, b in enumerate(LEARNING_STYLE_BIN_LIST)}

# Get keys and sizes for block options
BLOCK_KEYS = list(BLOCK_OPTIONS.keys())
BLOCK_SIZES = [len(BLOCK_OPTIONS[k]) for k in BLOCK_KEYS]

# Helper Functions
def one_hot(index: int, size: int) -> torch.Tensor:
    v = torch.zeros(size, device=DEVICE)
    v[index] = 1.0
    return v

def encode_conversation(conversation: List[str], max_length: int = 512) -> torch.Tensor:
    # Simple encoding: use length and word count features
    if not conversation:
        return torch.zeros(4, device=DEVICE)
    
    # Join all conversation turns
    full_text = " ".join(conversation)
    
    # Extract simple features
    features = torch.tensor([
        len(conversation),  # Number of turns
        len(full_text),     # Total character count
        len(full_text.split()),  # Total word count
        len(conversation) / max(len(full_text.split()), 1)  # Avg words per turn
    ], device=DEVICE, dtype=torch.float32)
    
    return features

def build_state_tensor(concept: str, style: str, conversation: List[str] = None) -> torch.Tensor:
    c_vec = one_hot(BIN2IDX_CONCEPT.get(concept, 0), len(CONCEPT_BIN_LIST))
    s_vec = one_hot(BIN2IDX_STYLE.get(style, 0), len(LEARNING_STYLE_BIN_LIST))
    
    # Add conversation encoding
    if conversation is None:
        conversation = []
    conv_vec = encode_conversation(conversation)
    
    return torch.cat([c_vec, s_vec, conv_vec])

def assemble_prompt(actions: Dict[str, int],
                    concept: str,
                    style: str,
                    last_q: str,
                    conversation: List[str] = None) -> str:
    filled_blocks = {
        k: BLOCK_OPTIONS[k][idx] for k, idx in actions.items()
    }
    
    # Format conversation history
    if conversation is None:
        conversation = []
    
    conversation_text = ""
    if conversation:
        conversation_text = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(conversation)])
    else:
        conversation_text = "No previous conversation history."
    
    # Add conversation parameter to the template
    filled_blocks['concept'] = concept
    filled_blocks['learning_style'] = style
    filled_blocks['last_question'] = last_q
    filled_blocks['conversation'] = conversation_text
    
    return PROMPT_TEMPLATE.format(**filled_blocks)

def call_llm_with_retry(system_prompt: str, user_prompt: str, retries: int = EVAL_RETRIES) -> Optional[str]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME_LLM,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE_LLM,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed: {e}. Retrying ({attempt+1}/{retries})...")
                time.sleep(2 ** attempt)  
            else:
                print(f"API call failed after {retries} attempts: {e}")
                return None

def get_tutoring_response(prompt: str) -> str:
    return call_llm_with_retry(
        system_prompt="You are a helpful AI tutor.",
        user_prompt=prompt
    ) or "Failed to generate response."

def extract_score_from_feedback(feedback: str) -> float:
    try:
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', feedback)
        
        # If we find numbers take the last one which is likely the final score
        if numbers:
            score = float(numbers[-1])
            return min(max(score, 0), 100)
        
        return 50.00  # Default score if no numbers found
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 50.00  # Default score if parsing fails

# Stricter evaluation function
def evaluate_response_strict(prompt: str, answer_text: str, baseline_response: str = None) -> float:
    # Create a more demanding evaluation prompt
    strict_evaluation_prompt = f"""
You are an EXTREMELY STRICT educational content evaluator. Be highly critical and look for ANY flaws.

Rubric (judge harshly):
{RUBRIC_TEXT}

Original Prompt:
{prompt}

Assistant Response to Evaluate:
{answer_text}

STRICT EVALUATION INSTRUCTIONS:
1. Be very critical - a score of 100 should be EXTREMELY rare (only for perfect responses)
2. Look for ANY unclear explanations, missing examples, poor structure, etc.
3. Deduct points heavily for:
   - Any confusing language
   - Missing learning style adaptation
   - Lack of engaging tone
   - Poor examples or analogies
   - Missing connections to related concepts
   - Inadequate feedback solicitation
4. A "good" response should score 60-75
5. An "excellent" response should score 76-85
6. Only truly exceptional responses deserve 86+
7. Responses with any significant flaws should score below 60

{f'BASELINE COMPARISON: Compare this response to this baseline: {baseline_response}' if baseline_response else ''}

Provide a detailed critique and then give a STRICT numerical score between 0-100. Remember: be harsh!
"""
    
    grader_feedback = call_llm_with_retry(
        system_prompt="You are an extremely strict educational content evaluator who rarely gives high scores.",
        user_prompt=strict_evaluation_prompt
    )
    
    if not grader_feedback:
        return 50.0
    
    score = extract_score_from_feedback(grader_feedback)
    
    # Apply reward scaling to prevent saturation
    scaled_score = score * REWARD_SCALE_FACTOR
    
    # Add small amount of noise to prevent identical rewards
    noise = random.gauss(0, REWARD_NOISE_STD)
    final_score = max(0, min(100, scaled_score + noise))
    
    return final_score

# Generate baseline response for comparison
def get_baseline_response(prompt: str) -> str:
    simple_prompt = f"Briefly explain the concept mentioned in this educational prompt: {prompt[:200]}..."
    return call_llm_with_retry(
        system_prompt="You are a basic tutor. Give a simple, short explanation.",
        user_prompt=simple_prompt
    ) or "Basic explanation."

# Keep original evaluate_response for compatibility, but make it stricter
def evaluate_response(prompt: str, answer_text: str) -> float:
    return evaluate_response_strict(prompt, answer_text)

def bin_concept(concept: str) -> str:
    concept_list_str = "\n".join([f"- {bin_name}" for bin_name in CONCEPT_BIN_LIST])
    
    response = call_llm_with_retry(
        system_prompt="You are an expert in educational psychology.",
        user_prompt=f"Please categorize the following concept into one of these bins:\n{concept_list_str}\n\nConcept: {concept}\n\nReturn only the bin name that best matches the concept.",
        retries=1
    )
    
    if response:
        # Find the best matching bin
        response_lower = response.lower()
        for bin_name in CONCEPT_BIN_LIST:
            if bin_name.lower() in response_lower:
                return bin_name
    
    return CONCEPT_BIN_LIST[-1] if CONCEPT_BIN_LIST else "Miscellaneous / Other"  # Default to last bin

def bin_learning_style(style: str) -> str:
    style_list_str = "\n".join([f"- {bin_name}" for bin_name in LEARNING_STYLE_BIN_LIST])
    
    response = call_llm_with_retry(
        system_prompt="You are an expert in educational psychology.",
        user_prompt=f"Please categorize the following learning style into one of these bins:\n{style_list_str}\n\nLearning style: {style}\n\nReturn only the bin name that best matches the learning style.",
        retries=1
    )
    
    if response:
        # Find the best matching bin
        response_lower = response.lower()
        for bin_name in LEARNING_STYLE_BIN_LIST:
            if bin_name.lower() in response_lower:
                return bin_name
    
    return LEARNING_STYLE_BIN_LIST[-1] if LEARNING_STYLE_BIN_LIST else "Multimodal/Flexible (catch-all)"  # Default to last bin

# Policy Network with dropout for better generalization
class PromptPolicy(nn.Module):    
    def __init__(self):
        super().__init__()
        # Calculate input dimension dynamically
        input_dim = len(CONCEPT_BIN_LIST) + len(LEARNING_STYLE_BIN_LIST) + 4  # +4 for conversation features
        
        # Add dropout for regularization
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout
        )
        
        # output head for each block type
        self.heads = nn.ModuleList(
            [nn.Linear(HIDDEN_SIZE, size) for size in BLOCK_SIZES]
        )
    
    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        logits = [head(h) for head in self.heads]
        return logits
    
    def get_actions(self, state: torch.Tensor, deterministic: bool = False, epsilon: float = 0.0):
        self.eval()  
        with torch.no_grad():
            logits = self(state)
            actions = {}
            log_probs = []
            entropies = []
            
            for i, (key, logit) in enumerate(zip(BLOCK_KEYS, logits)):
                # Add epsilon-greedy exploration
                if not deterministic and random.random() < epsilon:
                    # Random action for exploration
                    action_idx = torch.randint(0, len(logit), (1,)).item()
                    dist = Categorical(logits=logit)
                else:
                    # Policy-based action
                    dist = Categorical(logits=logit)
                    if deterministic:
                        action_idx = torch.argmax(logit)
                    else:
                        action_idx = dist.sample()
                
                actions[key] = int(action_idx)
                # FIXED: Proper tensor handling to avoid warning
                if isinstance(action_idx, torch.Tensor):
                    log_probs.append(dist.log_prob(action_idx))
                else:
                    log_probs.append(dist.log_prob(torch.tensor(action_idx, dtype=torch.long, device=DEVICE)))
                entropies.append(dist.entropy())
        
        return actions, log_probs, entropies

# Data Classes 
@dataclass
class EpisodeLog:
    episode: int
    reward: float
    raw_reward: float  
    actions: Dict[str, int]
    prompt: str
    response: str
    concept_bin: str
    learning_style_bin: str
    conversation_length: int
    epsilon: float   
    
    def to_dict(self):
        return asdict(self)

# Real-time logging function
def save_episode_log(log: EpisodeLog, filename: str = "training_logs.jsonl"):
    with open(filename, "a") as f:  # Use append mode
        f.write(json.dumps(log.to_dict()) + "\n")

# Training with exploration and better grading
def run_episode(policy: PromptPolicy,
                optimizer: torch.optim.Optimizer,
                baseline: float,
                concept: str,
                learning_style: str,
                conversation: List[str],
                episode: int,
                baseline_responses: Dict[str, str] = None) -> Tuple[float, EpisodeLog]:
    policy.train()
    
    # Calculate current exploration rate (epsilon decay)
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
    
    # Assume last question/comment in conversation is the student's latest input
    if conversation:
        STUDENT_QUESTION = conversation[-1]
    else:
        STUDENT_QUESTION = "No previous questions."

    # Bin the concept and learning style dynamically
    concept_bin = bin_concept(concept)
    learning_style_bin = bin_learning_style(learning_style)
    
    # State tensor from concept, learning style, and conversation
    state_tensor = build_state_tensor(concept_bin, learning_style_bin, conversation)

    # Get logits from policy network
    logits = policy(state_tensor)

    # FIXED: Sample actions with epsilon-greedy exploration and proper tensor handling
    actions = {}
    log_probs = []
    entropies = []
    
    for key, logit in zip(BLOCK_KEYS, logits):
        # Use epsilon-greedy for exploration
        if random.random() < epsilon:
            action_idx = random.randint(0, len(BLOCK_OPTIONS[key]) - 1)
            # Convert to tensor properly for log_prob calculation
            action_tensor = torch.tensor(action_idx, dtype=torch.long, device=DEVICE)
        else:
            dist = Categorical(logits=logit)
            action_tensor = dist.sample()  # This is already a tensor
            action_idx = action_tensor.item()  # Convert to Python int for storage
        
        actions[key] = action_idx  # Store as Python int
        
        # Create distribution for log_prob calculation
        dist = Categorical(logits=logit)
        log_probs.append(dist.log_prob(action_tensor))  # Use tensor directly
        entropies.append(dist.entropy())
    
    # Assemble prompt with selected actions
    prompt = assemble_prompt(actions, concept, learning_style, STUDENT_QUESTION, conversation)
    
    # Get tutoring response from LLM
    response = get_tutoring_response(prompt)
    
    # Get baseline response for comparison if enabled
    baseline_response = None
    if USE_COMPARATIVE_GRADING:
        if baseline_responses is None:
            baseline_responses = {}
        cache_key = f"{concept}_{learning_style}"
        if cache_key not in baseline_responses:
            baseline_responses[cache_key] = get_baseline_response(prompt)
        baseline_response = baseline_responses[cache_key]
    
    # Use strict evaluation with baseline comparison
    raw_reward = evaluate_response_strict(prompt, response, baseline_response)
    reward = raw_reward  # The function already applies scaling and noise
    
    # Calculate advantage (how much better/worse than baseline)
    advantage = reward - baseline
    
    # Calculate loss with stronger entropy bonus
    policy_loss = -(advantage) * torch.stack(log_probs).sum()
    entropy_loss = -ENTROPY_LAMBDA * torch.stack(entropies).sum()
    total_loss = policy_loss + entropy_loss
    
    # Update policy parameters
    optimizer.zero_grad()
    total_loss.backward()
    
    # Add gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Create log entry with more tracking
    log = EpisodeLog(
        episode=episode,
        reward=reward,
        raw_reward=raw_reward,
        actions=actions,
        prompt=prompt,
        response=response,
        concept_bin=concept_bin,
        learning_style_bin=learning_style_bin,
        conversation_length=len(conversation) if conversation else 0,
        epsilon=epsilon,
    )
    
    return reward, log

def save_logs(logs: List[EpisodeLog], filename: str = "training_logs.jsonl"):
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log.to_dict()) + "\n")
    print(f"Logs saved to {filename}")

# Logging with raw rewards and exploration tracking
def print_epoch_summary(episode: int, reward: float, recent_rewards: List[float], best_reward: float, epsilon: float, logs: List[EpisodeLog]):
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    
    # Calculate average raw reward (before scaling) for last few episodes
    recent_raw_rewards = [log.raw_reward for log in logs[-min(10, len(logs)):]]
    avg_raw_reward = sum(recent_raw_rewards) / len(recent_raw_rewards) if recent_raw_rewards else 0
    
    print(f"Episode {episode}/{N_EPISODES} | Reward: {reward:.2f} | Raw: {logs[-1].raw_reward:.2f} | "
          f"Avg: {avg_reward:.2f} | Raw Avg: {avg_raw_reward:.2f} | Best: {best_reward:.2f} | ε: {epsilon:.3f}")

def train_policy(concept: str = "linear algebra", 
                learning_style: str = "step by step with examples", 
                conversation: List[str] = None):
    torch.manual_seed(42)
    random.seed(42)
    
    if conversation is None:
        conversation = []
    
    policy = PromptPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    baseline = 0.0
    logs = []
    recent_rewards = []
    best_reward = float('-inf')
    best_episode = 0
    baseline_responses = {}  # Cache for baseline responses
    
    # NEW: Clear the log file at the start of training
    log_filename = "training_logs.jsonl"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    
    print(f"Starting training with {N_EPISODES} episodes...")
    print(f"Concept: {concept}")
    print(f"Learning Style: {learning_style}")
    print(f"Conversation length: {len(conversation)}")
    print(f"Exploration: {EPSILON_START} -> {EPSILON_END}")
    print(f"Reward scaling: {REWARD_SCALE_FACTOR}")
    print(f"Entropy weight: {ENTROPY_LAMBDA}")
    print(f"Real-time logging to: {log_filename}")
    print("="*60)
    
    for episode in range(1, N_EPISODES + 1):
        reward, log = run_episode(
            policy=policy,
            optimizer=optimizer,
            baseline=baseline,
            concept=concept,
            learning_style=learning_style,
            conversation=conversation,
            episode=episode,
            baseline_responses=baseline_responses
        )
        
        # Update baseline with exponential moving average
        baseline = BASELINE_BETA * baseline + (1 - BASELINE_BETA) * reward
        
        # Update tracking variables
        logs.append(log)
        recent_rewards.append(reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        
        # Track best model
        if reward > best_reward:
            best_reward = reward
            best_episode = episode
        
        # NEW: Save log immediately after each episode
        save_episode_log(log, log_filename)
        
        # Track logs and save every now and then with better logging
        if episode % LOG_INTERVAL == 0:
            print_epoch_summary(episode, reward, recent_rewards, best_reward, log.epsilon, logs)
    
    # Final save as backup (all logs at once)
    save_logs(logs, "training_logs_final.jsonl")
    
    print(f"\nTraining complete")
    print(f"Best reward: {best_reward:.2f} at episode {best_episode}")
    print(f"Final exploration rate: {logs[-1].epsilon:.3f}")
    print(f"Real-time logs saved to: {log_filename}")
    print(f"Final backup logs saved to: training_logs_final.jsonl")

    # Final evaluation
    print("\nFinal Evaluation:")
    
    # Bin the inputs for final evaluation
    concept_bin = bin_concept(concept)
    learning_style_bin = bin_learning_style(learning_style)
    
    # Get the final student question
    if conversation:
        STUDENT_QUESTION = conversation[-1]
    else:
        STUDENT_QUESTION = "No previous questions."
    
    actions, _, _ = policy.get_actions(
        build_state_tensor(concept_bin, learning_style_bin, conversation),
        deterministic=True
    )
    
    # Print best prompt configuration
    print("\nBest prompt configuration:")
    for key, idx in actions.items():
        print(f"{key}: {BLOCK_OPTIONS[key][idx]}")
    
    # Generate example prompt with best configuration
    best_prompt = assemble_prompt(actions, concept, learning_style, STUDENT_QUESTION, conversation)
    print("\nExample of best prompt template:")
    print(best_prompt)
    
    return policy, logs

def main():
    # Default parameters
    concept = "linear algebra"
    learning_style = "step by step with examples"
    conversation = []
    
    # If you want to run with binning step, uncomment the following lines:
    # concept = bin_concept(concept)
    # learning_style = bin_learning_style(learning_style)

    train_policy(concept, learning_style, conversation)

if __name__ == "__main__":
    main()