import os
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

DEVICE = "cpu"  # "cuda" if GPU is available
LR = 3e-4  
HIDDEN_SIZE = 32  
ENTROPY_LAMBDA = 1e-2  # exploration vs exploitation
BASELINE_BETA = 0.9  # exponential moving average factor for baseline
N_EPISODES = 500  
TEMPERATURE_LLM = 0  # deterministic LLM responses
MODEL_NAME_LLM = "gpt-4o-mini"  # LLM model for evaluation
EVAL_RETRIES = 3  # number of retry attempts for API calls
SAVE_INTERVAL = 10  
LOG_INTERVAL = 5 

# ---- CONCEPT AND LEARNING STYLE BINS ----
CONCEPT_BINS = ["Math", "Chemistry", "Biology", "History", "Finance"]
STYLE_BINS = ["Step-by-Step", "Visual Analogies", "Real-World Examples", "Socratic", "Quick Summary"]

# ---- PROMPT BLOCK OPTIONS ----
BLOCK_OPTIONS = {
    "core_role": [
        "Leading Industry Expert",
        "University Professor",
        "Cutting-Edge Researcher",
        "Patient Tutor",
        "Exam Coach",
    ],
    "pedagogical_strategy": [
        "Provide a worked example first, then break it down.",
        "Ask the student guiding questions before revealing the answer.",
        "Use an analogy drawn from real life.",
        "Present a concise definition, then extend with edge cases.",
    ],
    "answer_format": [
        "Markdown with bullet points and **bold** key terms.",
        "Numbered step-by-step list; one concept per line.",
        "Paragraph explanation followed by a boxed summary.",
        "Table comparing definitions vs. examples.",
    ],
    "memory_mode": [
        "Briefly recap previous answers in 2 sentences.",
        "Assume no prior context; start fresh.",
        "Only reference earlier misconceptions the student had.",
        "Embed a one-line summary of the last turn at the top.",
    ],
    "tone_style": [
        "Encouraging and upbeat.",
        "Formal and concise.",
        "Friendly peer-to-peer.",
        "Challenge the student with follow-up questions.",
    ],
}

# maps bins to indices for one-hot encoding
BIN2IDX_CONCEPT = {b: i for i, b in enumerate(CONCEPT_BINS)}
BIN2IDX_STYLE = {b: i for i, b in enumerate(STYLE_BINS)}

# get keys and sizes for block options
BLOCK_KEYS = list(BLOCK_OPTIONS.keys())
BLOCK_SIZES = [len(BLOCK_OPTIONS[k]) for k in BLOCK_KEYS]

# PROMPT
PROMPT_TEMPLATE = """
Imagine you are a {core_role}
teaching a student who prefers {learning_style}.

Topic focus: {concept_bin}
Student's question: "{last_question}"

{pedagogical_strategy}

When you answer, follow this format: {answer_format}

{memory_mode}

{tone_style}
""".strip()

STUDENT_QUESTION = "What is the rank of a matrix? How do I find it?"

# RUBRIC
RUBRIC_TEXT = """
1. clarity - is the explanation understandable & logically structured for the target learning style
score 1: contains multiple unclear sentences/poor structure
score 2: mostly understandable but includes 3+ confusing phrases/steps
score 3: understandable overall but has 1-2 unclear phrases or awkward transitions
score 4: clear except for 1 minor phrase/awkward transition
score 5: fully clear, logical, no confusing language

2. learning style fit - does the explanation align well with the requested learning style (ex. uses examples if example-based learner)
score 1: no visible attempt to tailor to the learning style
score 2: minimal tailoring, mentions style but does not change explanation
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
score 2: generic let me know style phrase without prompting further action
score 3: explicitly invites questions or feedback once
score 4: explicitly invites feedback and offers to adjust explanation
score 5: invites feedback, offers adjustment, encourages iterative interaction, and may also give multiple responses and ask which is best (think chatgpt lol)

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

# HELPERS
def one_hot(index: int, size: int) -> torch.Tensor:
    v = torch.zeros(size, device=DEVICE)
    v[index] = 1.0
    return v

def build_state_tensor(concept: str, style: str) -> torch.Tensor:
    c_vec = one_hot(BIN2IDX_CONCEPT.get(concept, 0), len(CONCEPT_BINS))
    s_vec = one_hot(BIN2IDX_STYLE.get(style, 0), len(STYLE_BINS))
    return torch.cat([c_vec, s_vec])  # Concatenate to form state vector

def assemble_prompt(actions: Dict[str, int],
                    concept: str,
                    style: str,
                    last_q: str) -> str:
    filled_blocks = {
        k: BLOCK_OPTIONS[k][idx] for k, idx in actions.items()
    }
    return PROMPT_TEMPLATE.format(
        concept_bin=concept,
        learning_style=style,
        last_question=last_q,
        **filled_blocks
    )

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
        
        # if we find numbers take the last one which is likely the final score
        if numbers:
            score = float(numbers[-1])
            return min(max(score, 0), 100)
        
        return 50.0  # default score if no numbers found
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 50.0  # Default score if parsing fails

def evaluate_response(answer_text: str) -> float:
    evaluation_prompt = f"{RUBRIC_TEXT}\n\nAssistant answer:\n{answer_text}\n\nPlease provide a final score between 0 and 100."
    
    grader_feedback = call_llm_with_retry(
        system_prompt="You are an expert educational content evaluator.",
        user_prompt=evaluation_prompt
    )
    
    if not grader_feedback:
        return 50.0  # Default score if evaluation fails
    
    score = extract_score_from_feedback(grader_feedback)
    return score

# POLICY NETWORK 
class PromptPolicy(nn.Module):    
    def __init__(self):
        super().__init__()
        input_dim = len(CONCEPT_BINS) + len(STYLE_BINS)
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        
        # output head for each block type
        self.heads = nn.ModuleList(
            [nn.Linear(HIDDEN_SIZE, size) for size in BLOCK_SIZES]
        )
    
    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        logits = [head(h) for head in self.heads]
        return logits
    
    def get_actions(self, state: torch.Tensor, deterministic: bool = False):
        self.eval()  
        with torch.no_grad():
            logits = self(state)
            actions = {}
            log_probs = []
            entropies = []
            
            for i, (key, logit) in enumerate(zip(BLOCK_KEYS, logits)):
                dist = Categorical(logits=logit)
                
                if deterministic:
                    # choose the most likely action <- deterministic approach remember
                    action_idx = torch.argmax(logit)
                else:
                    action_idx = dist.sample()
                
                actions[key] = int(action_idx)
                log_probs.append(dist.log_prob(action_idx))
                entropies.append(dist.entropy())
        
        return actions, log_probs, entropies

# DATA CLASSES FOR LOGGING 
@dataclass
class EpisodeLog:
    episode: int
    reward: float
    actions: Dict[str, int]
    prompt: str
    response: str
    
    def to_dict(self):
        return asdict(self)

# TRAINING 
def run_episode(policy: PromptPolicy,
                optimizer: torch.optim.Optimizer,
                baseline: float,
                concept_bin: str,
                learning_style: str,
                episode: int) -> Tuple[float, EpisodeLog]:
    policy.train()
    
    # state tensor from concept and learning style
    state_tensor = build_state_tensor(concept_bin, learning_style)
    
    # get logits from policy network
    logits = policy(state_tensor)
    
    # sample actions and calculate log probabilities and entropies
    actions = {}
    log_probs = []
    entropies = []
    
    for key, logit in zip(BLOCK_KEYS, logits):
        dist = Categorical(logits=logit)
        action_idx = dist.sample()
        actions[key] = int(action_idx)
        log_probs.append(dist.log_prob(action_idx))
        entropies.append(dist.entropy())
    
    # assemble prompt with selected actions
    prompt = assemble_prompt(actions, concept_bin, learning_style, STUDENT_QUESTION)
    
    # get tutoring response from LLM
    response = get_tutoring_response(prompt)
    
    # evaluate response quality
    reward = evaluate_response(response)
    
    # calculate advantage (how much better/worse than baseline)
    advantage = reward - baseline
    
    # calculate loss
    policy_loss = -(advantage) * torch.stack(log_probs).sum()
    entropy_loss = -ENTROPY_LAMBDA * torch.stack(entropies).sum()
    total_loss = policy_loss + entropy_loss
    
    # update policy parameters
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # create log entry
    log = EpisodeLog(
        episode=episode,
        reward=reward,
        actions=actions,
        prompt=prompt,
        response=response,
    )
    
    return reward, log

def save_model(policy: PromptPolicy, episode: int):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), f"checkpoints/policy_ep{episode}.pt")
    print(f"Model checkpoint saved at episode {episode}")

def save_logs(logs: List[EpisodeLog], filename: str = "training_logs.jsonl"):
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log.to_dict()) + "\n")
    print(f"Logs saved to {filename}")

def print_epoch_summary(episode: int, reward: float, recent_rewards: List[float], best_reward: float):
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    print(f"Episode {episode}/{N_EPISODES} | Reward: {reward:.2f} | "
          f"Avg last {len(recent_rewards)}: {avg_reward:.2f} | Best: {best_reward:.2f}")

def main():
    torch.manual_seed(42)
    random.seed(42)
    
    policy = PromptPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    baseline = 0.0
    logs = []
    recent_rewards = []
    best_reward = float('-inf')
    best_episode = 0
    
    concept_bin = "Math"
    learning_style = "Step-by-Step"
    
    print(f"Starting training with {N_EPISODES} episodes...")
    print(f"Concept: {concept_bin}, Learning Style: {learning_style}")
    
    for episode in range(1, N_EPISODES + 1):
        reward, log = run_episode(
            policy=policy,
            optimizer=optimizer,
            baseline=baseline,
            concept_bin=concept_bin,
            learning_style=learning_style,
            episode=episode
        )
        
        # update baseline with exponential moving average
        baseline = BASELINE_BETA * baseline + (1 - BASELINE_BETA) * reward
        
        # update tracking variables
        logs.append(log)
        recent_rewards.append(reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        
        # track best model
        if reward > best_reward:
            best_reward = reward
            best_episode = episode
            save_model(policy, episode)
        
        # track logs and save every now and then 
        if episode % LOG_INTERVAL == 0:
            print_epoch_summary(episode, reward, recent_rewards, best_reward)
        
        if episode % SAVE_INTERVAL == 0:
            save_model(policy, episode)
            save_logs(logs)
    
    # final model 
    save_model(policy, N_EPISODES)
    save_logs(logs)
    
    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.2f} at episode {best_episode}")
    
    # final evaluation with best policy
    print("\nEvaluating best policy...")
    policy.load_state_dict(torch.load(f"checkpoints/policy_ep{best_episode}.pt"))
    actions, _, _ = policy.get_actions(
        build_state_tensor(concept_bin, learning_style),
        deterministic=True
    )
    
    # print best prompt configuration
    print("\nBest prompt configuration:")
    for key, idx in actions.items():
        print(f"{key}: {BLOCK_OPTIONS[key][idx]}")
    
    # generate example prompt with best configuration
    best_prompt = assemble_prompt(actions, concept_bin, learning_style, STUDENT_QUESTION)
    print("\nExample of best prompt template:")
    print(best_prompt)

if __name__ == "__main__":
    main()