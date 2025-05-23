import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import os
import json
import time
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Constants
DEVICE = "cpu"
LR = 3e-4
HIDDEN_SIZE = 32
ENTROPY_LAMBDA = 1e-2
BASELINE_BETA = 0.9
N_EPISODES = 100
TEMPERATURE_LLM = 0
MODEL_NAME_LLM = "gemini-pro"
EVAL_RETRIES = 3
SAVE_INTERVAL = 10
LOG_INTERVAL = 5

# Rubric for evaluation
RUBRIC_TEXT = """
1. clarity - is the explanation understandable & logically structured for the target learning style
score 1: contains multiple unclear sentences/poor structure
score 2: mostly understandable but includes 3+ confusing phrases/steps
score 3: understandable overall but has 1-2 unclear phrases or awkward transitions
score 4: clear except for 1 minor phrase/awkward transition
score 5: fully clear, logical, no confusing language

2. learning style fit - does the explanation align well with the requested learning style (ex. uses examples if example-based learner)
score 1: no visible attempt to tailor to the learning style
score 2: minimal tailoring, mentions style but doesn't change explanation
score 3: partial adaptation, includes 1 correct technique for style
score 4: strong adaptation, includes 2+ correct techniques for style
score 5: fully tailored, uses correct teaching approach throughout

3. accuracy - is the output correct
score 1: contains major factual error(s) or wrong steps
score 2: contains 2+ minor factual errors or 1 medium error
score 3: contains 1 minor factual error
score 4: Fully accurate with only a nitpicky issue
score 5: Completely accurate, correct, nothing to correct

4. engagement - does the explanation use phrasing or style that keeps a student interested
score 1: monotone, generic, no attention to tone
score 2: slightly engaging phrasing in 1-2 spots
score 3: engaging tone in some parts but inconsistent
score 4: consistently uses friendly/interesting phrasing
score 5: very engaging tone throughout, holds attention

5. use of examples - are examples included and do they support understanding (esp for ex-based learners)
score 1: no examples present when learning style requested them
score 2: example included but irrelevant or unclear
score 3: example relevant but underdeveloped/too simple
score 4: example relevant and well-explained
score 5: examples highly relevant, well-explained, directly clarify concept

6. effectiveness of recap - for guides with a recap block, does it summarize key ideas effectively
score 1: no recap included
score 2: recap included but missing key ideas
score 3: recap includes most ideas but vague
score 4: recap covers all main ideas clearly
score 5: recap clearly, concisely, and completely summarizes prior content

7. anticipating misconceptions - does the guide address likely student misunderstandings
score 1: does not address any misconceptions
score 2: mentions misconception but w/o explanation
score 3: addresses 1 key misconception
score 4: addresses 2+ key misconceptions
score 5: proactively addresses misconceptions and clarifies them effectively

8. motivational boost - does the explanation end with an encouraging/affirming statement
score 1: no motivational or affirming language
score 2: generic closing with mild positivity
score 3: clear affirming statement included once
score 4: closing includes affirming statement and encouragement to continue
score 5: closing includes affirming statement, encouragement, and suggestion for next steps

9. user feedback - does it consistently ask for and learn from user feedback on the output
score 1: no invitation for feedback or questions
score 2: generic "let me know" style phrase without prompting further action
score 3: explicitly invites questions or feedback once
score 4: explicitly invites feedback and offers to adjust explanation
score 5: invites feedback, offers adjustment, encourages iterative interaction, and may also give multiple responses and ask which is best

10. formatting - how well does formatting improve readability (ex. bullet points, bold key terms, section headers)
score 1: plain paragraph w/ no formatting
score 2: 1 formatting element used (ex. 1 header only)
score 3: 2 formatting elements used inconsistently
score 4: 2-3 formatting elements used consistently
score 5: 3+ formatting elements used effectively and consistently to improve readability

weights:
clarity: 0.15
learning style fit: 0.20
accuracy: 0.15
engagement: 0.10
use of examples: 0.05
effectiveness of recap: 0.05
anticipating misconceptions: 0.05
motivational boost: 0.05
user feedback: 0.10
formatting: 0.10

how to use the metrics: each metric is scored from 1 to 5 using the above criteria, multiply each score by its weight and sum for a final evaluation score
Please output a numeric score between 0 and 100 for the overall quality of the answer.
""".strip()

# Prompt template
PROMPT_TEMPLATE = """
Imagine you are a {teacher_role} teaching a student about {concept}. The student is a {learning_style} learner. 
1. begin with a concise explanation of {concept}, structured logically for clarity.
2. provide at least one real-world example illustrating the concept.
3. use {theoretical_strategy} techniques (ex. analogies, asking questions).
4. present your answer in {answer_format} (ex. numbered steps, bullet list).
5. {memory_mode}
6. {tone_style}
7. include a brief recap: "In summary, …".
8. at the end, offer motivation: "you've got this, keep going!".
9. encourage feedback: "does this make sense? lmk if you'd like more examples or a diff approach."
Write everything in plain text, no emojis or symbols.
""".strip()

# bin defns
# bin 1: concepts
CONCEPTS: List[str] = [
    # Math
    "Algebra", "Calculus", "Linear Algebra", "Probability", "Statistics", "Differential Equations",
    # CS
    "Data Structures & Algorithms", "Databases", "Machine Learning", "Artificial Intelligence", "Operating Systems",
    # Physics
    "Mechanics", "Electromagnetism", "Thermodynamics", "Quantum Physics",
    # Chemistry
    "Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Biochemistry",
    # Biology
    "Cell Biology", "Genetics", "Ecology", "Neuroscience",
    # Economics
    "Microeconomics", "Macroeconomics", "Finance", "Game Theory",
    # Engineering
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Software Engineering",
    # Humanities
    "History", "Literature", "Philosophy", "Languages",
    # Social Sciences
    "Psychology", "Sociology", "Anthropology",
    # Business & Marketing
    "Accounting", "Business Strategy", "Analytics", "Pitching", "Entrepreneurship",
    # Health & Medicine
    "Anatomy", "Physiology", "Pharmacology",
    # Art & Design
    "Drawing", "Graphic Design", "UI/UX Design",
    # Writing & Communication
    "Essay Writing", "Technical Writing", "Presentation Skills", "Debate",
    # Other
    "Ethics", "Environmental Science", "Law", "Music Theory"
]

# bin 2: learning styles
LEARNING_STYLES: List[str] = [
    "Visual", "Auditory", "Example-Based", "Analogy-Based", "Step-by-Step",
    "Asking Questions", "Practice-First", "Contextual/Real-World",
    "Recap-Focused", "Interactive", "Concise", "Storytelling", "Combining Methods"
]

# action space defns
TEACHER_ROLES: List[str] = [
    "Experienced college professor with 10 years teaching experience",
    "Friendly peer tutor with practical insights",
    "Industry expert with real-world applications",
    "Patient mentor specializing in beginners",
    "Enthusiastic science communicator"
]

THEORETICAL_STRATEGIES: List[str] = [
    "Use of analogies and metaphors",
    "Asking probing questions to guide thinking",
    "Providing mnemonics and memory aids",
    "Relating to prior knowledge and experiences",
    "Breaking down complex ideas into simple steps"
]

ANSWER_FORMATS: List[str] = [
    "Bulleted list with key points",
    "Numbered step-by-step explanation",
    "Detailed paragraph with clear sections",
    "Q&A format with common questions",
    "Concept map description with relationships"
]

MEMORY_MODES: List[str] = [
    "Briefly recap previous answers in 2 sentences",
    "Assume no prior context; start fresh",
    "Only reference earlier misconceptions the student had",
    "Embed a one-line summary of the last turn at the top"
]

TONE_STYLES: List[str] = [
    "Encouraging and upbeat",
    "Formal and concise",
    "Friendly peer-to-peer",
    "Challenge the student with follow-up questions"
]

# mappings for NN input/output
concept_to_idx: Dict[str, int] = {concept: i for i, concept in enumerate(CONCEPTS)}
idx_to_concept: Dict[int, str] = {i: concept for i, concept in enumerate(CONCEPTS)}

style_to_idx: Dict[str, int] = {style: i for i, style in enumerate(LEARNING_STYLES)}
idx_to_style: Dict[int, str] = {i: style for i, style in enumerate(LEARNING_STYLES)}

# placeholder rubric eval
# this fxn will simulate the process of generating a response with an LLM using the chosen parameters and then scoring it with the rubric
# for now, it's hardcoded and will return a somewhat random score, but ideally, it should favor certain combos to guide the learning

def calculate_rubric_score(concept_idx: int, style_idx: int, role_idx: int, strategy_idx: int, 
                         format_idx: int, memory_idx: int, tone_idx: int) -> float:
    """
    Calculate rubric score between 0 and 100.
    The actual implementation would involve:
    1. Formatting the prompt template with the chosen parameters.
    2. Sending the prompt to an LLM (e.g., Chelle AI).
    3. Evaluating the LLM's response using the detailed rubric.
    """
    # introduce some bias for demonstration:
    score: float = 60.0  # base score (3.0 * 20 to scale to 0-100)

    if LEARNING_STYLES[style_idx] == "Visual" and ANSWER_FORMATS[format_idx] == "Concept map description":
        score += 30.0  # 1.5 * 20
    if LEARNING_STYLES[style_idx] == "Example-Based" and THEORETICAL_STRATEGIES[strategy_idx] == "Use of analogies":
        score += 20.0  # 1.0 * 20
    if CONCEPTS[concept_idx] == "Algebra" and LEARNING_STYLES[style_idx] == "Step-by-Step":
        score += 20.0  # 1.0 * 20
    if TEACHER_ROLES[role_idx] == "Patient mentor for beginners":
        score += 10.0  # 0.5 * 20
    if MEMORY_MODES[memory_idx] == "Briefly recap previous answers in 2 sentences":
        score += 10.0  # 0.5 * 20
    if TONE_STYLES[tone_idx] == "Encouraging and upbeat":
        score += 10.0  # 0.5 * 20

    # add some randomness
    score += np.random.uniform(-10.0, 10.0)  # -0.5 to 0.5 * 20
    
    # make sure score is within 0-100 range
    return max(0.0, min(100.0, score))

# NN defn
class PromptPolicyNetwork(nn.Module):
    def __init__(self, num_concepts: int, num_styles: int, num_roles: int, num_strategies: int, 
                 num_formats: int, num_memory_modes: int, num_tone_styles: int, 
                 embedding_dim: int = 32, hidden_dim: int = 64):
        super(PromptPolicyNetwork, self).__init__()
        self.concept_embedding = nn.Embedding(num_concepts, embedding_dim)
        self.style_embedding = nn.Embedding(num_styles, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc_role = nn.Linear(hidden_dim, num_roles)
        self.fc_strategy = nn.Linear(hidden_dim, num_strategies)
        self.fc_format = nn.Linear(hidden_dim, num_formats)
        self.fc_memory = nn.Linear(hidden_dim, num_memory_modes)
        self.fc_tone = nn.Linear(hidden_dim, num_tone_styles)

    def forward(self, concept_idx: torch.Tensor, style_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        concept_embed = self.concept_embedding(concept_idx)
        style_embed = self.style_embedding(style_idx)

        # concatenate embeddings
        x = torch.cat((concept_embed, style_embed), dim=-1)
        
        x = F.relu(self.fc1(x))

        # output logits for each action dimension
        role_logits = self.fc_role(x)
        strategy_logits = self.fc_strategy(x)
        format_logits = self.fc_format(x)
        memory_logits = self.fc_memory(x)
        tone_logits = self.fc_tone(x)

        return role_logits, strategy_logits, format_logits, memory_logits, tone_logits

# fxn to select 20 unique (concept, style) permutations
def get_target_permutations(num_permutations: int = 20) -> List[Tuple[str, str]]:
    all_pairs: List[Tuple[str, str]] = []
    for concept_name in CONCEPTS:
        for style_name in LEARNING_STYLES:
            all_pairs.append((concept_name, style_name))
    
    if len(all_pairs) < num_permutations:
        print(f"Warning: Only {len(all_pairs)} unique (concept, style) pairs available. Using all of them.")
        return all_pairs
    
    # shuffle for variety if we have more than num_permutations, then pick
    np.random.shuffle(all_pairs)
    return all_pairs[:num_permutations]

@dataclass
class EpisodeLog:
    episode: int
    reward: float
    actions: Dict[str, int]
    prompt: str
    response: str
    
    def to_dict(self):
        return asdict(self)

def call_llm_with_retry(prompt: str, retries: int = EVAL_RETRIES) -> Optional[str]:
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(MODEL_NAME_LLM)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"API call failed: {e}. Retrying ({attempt+1}/{retries})...")
                time.sleep(2 ** attempt)
            else:
                print(f"API call failed after {retries} attempts: {e}")
                return None

def get_tutoring_response(prompt: str) -> str:
    return call_llm_with_retry(prompt) or "Failed to generate response."

def evaluate_response(answer_text: str) -> float:
    evaluation_prompt = f"{RUBRIC_TEXT}\n\nAssistant answer:\n{answer_text}\n\nPlease provide a final score between 0 and 100."
    
    grader_feedback = call_llm_with_retry(evaluation_prompt)
    
    if not grader_feedback:
        return 50.0  # Default score if evaluation fails
    
    try:
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', grader_feedback)
        if numbers:
            score = float(numbers[-1])
            return min(max(score, 0), 100)  # Ensure score is between 0 and 100
        return 50.0
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 50.0

def assemble_prompt(actions: Dict[str, int], concept: str, style: str) -> str:
    filled_blocks = {
        "teacher_role": TEACHER_ROLES[actions["role"]],
        "concept": concept,
        "learning_style": style,
        "theoretical_strategy": THEORETICAL_STRATEGIES[actions["strategy"]],
        "answer_format": ANSWER_FORMATS[actions["format"]],
        "memory_mode": MEMORY_MODES[actions["memory"]],
        "tone_style": TONE_STYLES[actions["tone"]]
    }
    return PROMPT_TEMPLATE.format(**filled_blocks)

def save_model(policy: PromptPolicyNetwork, episode: int):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), f"checkpoints/policy_ep{episode}.pt")
    print(f"Model checkpoint saved at episode {episode}")

def save_logs(logs: List[EpisodeLog], filename: str = "training_logs.jsonl"):
    # Create a backup of existing logs if file exists
    if os.path.exists(filename):
        backup_filename = f"{filename}.backup"
        os.rename(filename, backup_filename)
    
    # Write all logs to the file
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log.to_dict()) + "\n")
    print(f"Logs saved to {filename}")

def print_epoch_summary(episode: int, reward: float, recent_rewards: List[float], best_reward: float):
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    print(f"Episode {episode}/{N_EPISODES} | Reward: {reward:.2f} | "
          f"Avg last {len(recent_rewards)}: {avg_reward:.2f} | Best: {best_reward:.2f}")

def train_rl_agent():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize policy network
    net = PromptPolicyNetwork(
        num_concepts=len(CONCEPTS),
        num_styles=len(LEARNING_STYLES),
        num_roles=len(TEACHER_ROLES),
        num_strategies=len(THEORETICAL_STRATEGIES),
        num_formats=len(ANSWER_FORMATS),
        num_memory_modes=len(MEMORY_MODES),
        num_tone_styles=len(TONE_STYLES),
        embedding_dim=16,
        hidden_dim=32
    ).to(DEVICE)
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    baseline = 0.0
    logs: List[EpisodeLog] = []
    recent_rewards: List[float] = []
    best_reward = float('-inf')
    best_episode = 0
    
    target_permutations = get_target_permutations(20)
    total_episodes = len(target_permutations) * N_EPISODES
    current_episode = 0
    
    print(f"Starting training with {N_EPISODES} episodes per permutation")
    print(f"Total training episodes: {total_episodes}")
    print(f"Training on {len(target_permutations)} concept-style pairs:")
    for i, (concept, style) in enumerate(target_permutations, 1):
        print(f"{i}. Concept: {concept}, Style: {style}")
    
    for perm_idx, (concept_name, style_name) in enumerate(target_permutations, 1):
        print(f"\nStarting permutation {perm_idx}/{len(target_permutations)}")
        print(f"Concept: {concept_name}, Style: {style_name}")
        
        current_concept_idx = concept_to_idx[concept_name]
        current_style_idx = style_to_idx[style_name]
        
        # Initialize tensors for this permutation
        concept_tensor = torch.LongTensor([current_concept_idx]).to(DEVICE)
        style_tensor = torch.LongTensor([current_style_idx]).to(DEVICE)
        
        for episode in range(1, N_EPISODES + 1):
            current_episode += 1
            
            # Get actions from policy
            role_logits, strategy_logits, format_logits, memory_logits, tone_logits = net(concept_tensor, style_tensor)
            
            role_probs = F.softmax(role_logits, dim=-1)
            strategy_probs = F.softmax(strategy_logits, dim=-1)
            format_probs = F.softmax(format_logits, dim=-1)
            memory_probs = F.softmax(memory_logits, dim=-1)
            tone_probs = F.softmax(tone_logits, dim=-1)
            
            role_dist = torch.distributions.Categorical(role_probs)
            strategy_dist = torch.distributions.Categorical(strategy_probs)
            format_dist = torch.distributions.Categorical(format_probs)
            memory_dist = torch.distributions.Categorical(memory_probs)
            tone_dist = torch.distributions.Categorical(tone_probs)
            
            chosen_role_idx = role_dist.sample()
            chosen_strategy_idx = strategy_dist.sample()
            chosen_format_idx = format_dist.sample()
            chosen_memory_idx = memory_dist.sample()
            chosen_tone_idx = tone_dist.sample()
            
            # Calculate log probabilities and entropies
            log_prob_role = role_dist.log_prob(chosen_role_idx)
            log_prob_strategy = strategy_dist.log_prob(chosen_strategy_idx)
            log_prob_format = format_dist.log_prob(chosen_format_idx)
            log_prob_memory = memory_dist.log_prob(chosen_memory_idx)
            log_prob_tone = tone_dist.log_prob(chosen_tone_idx)
            
            total_log_prob = log_prob_role + log_prob_strategy + log_prob_format + log_prob_memory + log_prob_tone
            
            # Get reward
            reward = calculate_rubric_score(
                current_concept_idx, current_style_idx,
                int(chosen_role_idx.item()), int(chosen_strategy_idx.item()), 
                int(chosen_format_idx.item()), int(chosen_memory_idx.item()),
                int(chosen_tone_idx.item())
            )
            
            # Update baseline
            baseline = BASELINE_BETA * baseline + (1 - BASELINE_BETA) * reward
            
            # Calculate loss
            advantage = reward - baseline
            policy_loss = -(advantage) * total_log_prob
            entropy_loss = -ENTROPY_LAMBDA * (
                role_dist.entropy() + strategy_dist.entropy() + 
                format_dist.entropy() + memory_dist.entropy() + 
                tone_dist.entropy()
            )
            total_loss = policy_loss + entropy_loss
            
            # Update policy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Create log entry
            actions = {
                "role": int(chosen_role_idx.item()),
                "strategy": int(chosen_strategy_idx.item()),
                "format": int(chosen_format_idx.item()),
                "memory": int(chosen_memory_idx.item()),
                "tone": int(chosen_tone_idx.item())
            }
            
            log = EpisodeLog(
                episode=current_episode,
                reward=reward,
                actions=actions,
                prompt=f"Concept: {concept_name}, Style: {style_name}",
                response=f"Role: {TEACHER_ROLES[actions['role']]}, Strategy: {THEORETICAL_STRATEGIES[actions['strategy']]}, Format: {ANSWER_FORMATS[actions['format']]}, Memory: {MEMORY_MODES[actions['memory']]}, Tone: {TONE_STYLES[actions['tone']]}"
            )
            
            logs.append(log)
            recent_rewards.append(reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            
            # Track best model
            if reward > best_reward:
                best_reward = reward
                best_episode = current_episode
                save_model(net, current_episode)
            
            # Print progress
            if episode % LOG_INTERVAL == 0:
                print_epoch_summary(current_episode, reward, recent_rewards, best_reward)
            
            # Save checkpoints
            if episode % SAVE_INTERVAL == 0:
                save_model(net, current_episode)
                save_logs(logs)
        
        # Save final model and logs for this permutation
        save_model(net, current_episode)
        save_logs(logs)
        
        print(f"\nCompleted permutation {perm_idx}/{len(target_permutations)}")
        print(f"Best reward: {best_reward:.2f} at episode {best_episode}")
        
        # Print best configuration
        with torch.no_grad():
            role_logits, strategy_logits, format_logits, memory_logits, tone_logits = net(concept_tensor, style_tensor)
            best_role_idx = int(torch.argmax(role_logits, dim=-1).item())
            best_strategy_idx = int(torch.argmax(strategy_logits, dim=-1).item())
            best_format_idx = int(torch.argmax(format_logits, dim=-1).item())
            best_memory_idx = int(torch.argmax(memory_logits, dim=-1).item())
            best_tone_idx = int(torch.argmax(tone_logits, dim=-1).item())
            
            print("\nBest configuration for this permutation:")
            print(f"Role: {TEACHER_ROLES[best_role_idx]}")
            print(f"Strategy: {THEORETICAL_STRATEGIES[best_strategy_idx]}")
            print(f"Format: {ANSWER_FORMATS[best_format_idx]}")
            print(f"Memory: {MEMORY_MODES[best_memory_idx]}")
            print(f"Tone: {TONE_STYLES[best_tone_idx]}")
    
    print("\nTraining complete!")
    print(f"Total episodes completed: {current_episode}")
    print(f"Best overall reward: {best_reward:.2f} at episode {best_episode}")

if __name__ == "__main__":
    train_rl_agent() 