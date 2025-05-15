import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any

# bin defns
# bin 1: concepts
# simplified for initial implementation; expand with full list
CONCEPTS: List[str] = [
    "Algebra", "Calculus", "Data Structures", "Machine Learning",
    "Classical Mechanics", "Organic Chemistry", "Cell Biology", "Microeconomics",
    "Software Engineering", "History of Art", "Essay Writing", "Environmental Science"
]

# bin 2: learning styles
# simplified for initial implementation; expand with full list
LEARNING_STYLES: List[str] = [
    "Visual", "Auditory", "Example-Based", "Analogy-Based", "Step-by-Step",
    "Asking Questions", "Practice-First", "Contextual/Real-World",
    "Recap-Focused", "Interactive", "Concise", "Storytelling"
]

# action space defns
# these are the parameters the RL agent will learn to choose
TEACHER_ROLES: List[str] = [
    "Experienced college professor", "Friendly peer tutor", "Industry expert with practical insights",
    "Patient mentor for beginners", "Enthusiastic science communicator"
]

THEORETICAL_STRATEGIES: List[str] = [
    "Use of analogies", "Asking probing questions", "Providing mnemonics",
    "Relating to prior knowledge", "Breaking down complex ideas simply"
]

ANSWER_FORMATS: List[str] = [
    "Bulleted list", "Numbered steps", "Detailed paragraph",
    "Q&A format", "Concept map description"
]

# mappings for NN input/output
concept_to_idx: Dict[str, int] = {concept: i for i, concept in enumerate(CONCEPTS)}
idx_to_concept: Dict[int, str] = {i: concept for i, concept in enumerate(CONCEPTS)}

style_to_idx: Dict[str, int] = {style: i for i, style in enumerate(LEARNING_STYLES)}
idx_to_style: Dict[int, str] = {i: style for i, style in enumerate(LEARNING_STYLES)}

# placeholder rubric eval
# this fxn will simulate the process of generating a response with an LLM using the chosen parameters and then scoring it with the rubric
# for now, it's hardcoded and will return a somewhat random score, but ideally, it should favor certain combos to guide the learning

def calculate_rubric_score(concept_idx: int, style_idx: int, role_idx: int, strategy_idx: int, format_idx: int) -> float:
    """
    Placeholder for rubric scoring.
    Returns a score between 1 and 5.
    The actual implementation would involve:
    1. Formatting the prompt template with the chosen parameters.
    2. Sending the prompt to an LLM (e.g., Chelle AI).
    3. Evaluating the LLM's response using the detailed rubric.
    """
    # introduce some bias for demonstration:
    # ex. 'Visual' learners might benefit from 'Concept map description'
    # 'Algebra' with 'Step-by-Step' might be good
    score: float = 3.0  # base score

    if LEARNING_STYLES[style_idx] == "Visual" and ANSWER_FORMATS[format_idx] == "Concept map description":
        score += 1.5
    if LEARNING_STYLES[style_idx] == "Example-Based" and THEORETICAL_STRATEGIES[strategy_idx] == "Use of analogies":
        score += 1.0
    if CONCEPTS[concept_idx] == "Algebra" and LEARNING_STYLES[style_idx] == "Step-by-Step":
        score += 1.0
    if TEACHER_ROLES[role_idx] == "Patient mentor for beginners":
        score += 0.5

    # add some randomness
    score += np.random.uniform(-0.5, 0.5)
    
    # make sure score is within 1-5 range
    return max(1.0, min(5.0, score))

# NN defn
class PromptPolicyNetwork(nn.Module):
    def __init__(self, num_concepts: int, num_styles: int, num_roles: int, num_strategies: int, num_formats: int, embedding_dim: int =32, hidden_dim: int =64):
        super(PromptPolicyNetwork, self).__init__()
        self.concept_embedding = nn.Embedding(num_concepts, embedding_dim)
        self.style_embedding = nn.Embedding(num_styles, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc_role = nn.Linear(hidden_dim, num_roles)
        self.fc_strategy = nn.Linear(hidden_dim, num_strategies)
        self.fc_format = nn.Linear(hidden_dim, num_formats)

    def forward(self, concept_idx: torch.Tensor, style_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concept_embed = self.concept_embedding(concept_idx)
        style_embed = self.style_embedding(style_idx)

        # concatenate embeddings
        x = torch.cat((concept_embed, style_embed), dim=-1)
        
        x = F.relu(self.fc1(x))

        # output logits for each action dimension
        role_logits = self.fc_role(x)
        strategy_logits = self.fc_strategy(x)
        format_logits = self.fc_format(x)

        return role_logits, strategy_logits, format_logits

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

# main RL training fxn
def train_rl_agent():
    # hyperparameters
    episodes_per_permutation: int = 200 # reduced for faster demonstration per permutation
    learning_rate: float = 0.001
    embedding_dim: int = 16
    hidden_dim: int = 32
    num_target_permutations: int = 20

    target_permutations = get_target_permutations(num_target_permutations)
    
    print(f"Identified {len(target_permutations)} target permutations for training.")
    for i, (cn, sn) in enumerate(target_permutations):
        print(f"Permutation {i+1}/{len(target_permutations)}: Concept='{cn}', Style='{sn}'")

    trained_models_info: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for perm_idx, (concept_name, style_name) in enumerate(target_permutations):
        print(f"\n--- Training for Permutation {perm_idx + 1}/{len(target_permutations)}: Concept: {concept_name}, Style: {style_name} ---")

        current_concept_idx: int = concept_to_idx[concept_name]
        current_style_idx: int = style_to_idx[style_name]

        # init a new network and optimizer for each permutation
        net = PromptPolicyNetwork(
            num_concepts=len(CONCEPTS),
            num_styles=len(LEARNING_STYLES),
            num_roles=len(TEACHER_ROLES),
            num_strategies=len(THEORETICAL_STRATEGIES),
            num_formats=len(ANSWER_FORMATS),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        for episode in range(episodes_per_permutation):
            concept_tensor = torch.LongTensor([current_concept_idx])
            style_tensor = torch.LongTensor([current_style_idx])

            role_logits, strategy_logits, format_logits = net(concept_tensor, style_tensor)

            role_probs = F.softmax(role_logits, dim=-1)
            strategy_probs = F.softmax(strategy_logits, dim=-1)
            format_probs = F.softmax(format_logits, dim=-1)

            role_dist = torch.distributions.Categorical(role_probs)
            strategy_dist = torch.distributions.Categorical(strategy_probs)
            format_dist = torch.distributions.Categorical(format_probs)

            chosen_role_idx = role_dist.sample()
            chosen_strategy_idx = strategy_dist.sample()
            chosen_format_idx = format_dist.sample()
            
            log_prob_role = role_dist.log_prob(chosen_role_idx)
            log_prob_strategy = strategy_dist.log_prob(chosen_strategy_idx)
            log_prob_format = format_dist.log_prob(chosen_format_idx)
            
            total_log_prob = log_prob_role + log_prob_strategy + log_prob_format

            reward = calculate_rubric_score(
                current_concept_idx, current_style_idx,
                int(chosen_role_idx.item()), int(chosen_strategy_idx.item()), int(chosen_format_idx.item())
            )
            
            current_loss = -total_log_prob * reward # standard reinforce loss

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            if (episode + 1) % (episodes_per_permutation // 2) == 0 or episode == episodes_per_permutation -1 : # Print a couple of times per permutation
                 print(f"  Episode {episode + 1}/{episodes_per_permutation}, Reward: {reward:.2f}, Loss: {current_loss.item():.2f}")

        # after training for this permutation, get the best actions
        with torch.no_grad():
            concept_tensor = torch.LongTensor([current_concept_idx])
            style_tensor = torch.LongTensor([current_style_idx])
            role_logits, strategy_logits, format_logits = net(concept_tensor, style_tensor)
            
            best_role_idx: int = int(torch.argmax(role_logits, dim=-1).item())
            best_strategy_idx: int = int(torch.argmax(strategy_logits, dim=-1).item())
            best_format_idx: int = int(torch.argmax(format_logits, dim=-1).item())

            trained_models_info[(concept_name, style_name)] = {
                "role": TEACHER_ROLES[best_role_idx],
                "strategy": THEORETICAL_STRATEGIES[best_strategy_idx],
                "format": ANSWER_FORMATS[best_format_idx],
                "final_reward_check": calculate_rubric_score( # check score with best choice
                    current_concept_idx, current_style_idx,
                    best_role_idx, best_strategy_idx, best_format_idx
                )
            }
            print(f"  Optimal parameters found for {concept_name} & {style_name}:")
            print(f"    Teacher Role: {TEACHER_ROLES[best_role_idx]}")
            print(f"    Theoretical Strategy: {THEORETICAL_STRATEGIES[best_strategy_idx]}")
            print(f"    Answer Format: {ANSWER_FORMATS[best_format_idx]}")
            print(f"    (Deterministic Check Score: {trained_models_info[(concept_name, style_name)]['final_reward_check']:.2f})")


    print("\n--- Summary of Optimal Parameters for All Trained Permutations ---")
    for (c_name, s_name), params in trained_models_info.items():
        print(f"Concept: {c_name}, Style: {s_name}")
        print(f"  Optimal Role: {params['role']}")
        print(f"  Optimal Strategy: {params['strategy']}")
        print(f"  Optimal Format: {params['format']}")
        print(f"  (Final Check Score: {params['final_reward_check']:.2f})")
        print("-" * 20)
    
    print("\nTraining finished.")

if __name__ == "__main__":
    train_rl_agent() 