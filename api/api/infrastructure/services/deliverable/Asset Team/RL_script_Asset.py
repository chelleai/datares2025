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
from Bins_Asset import CONCEPT_BINS, CONTEXT_BINS
from Block_options_Asset import BLOCK_OPTIONS
from Prompt_template_Asset import PROMPT_TEMPLATE
from Rubric_Asset import EVALUATION_METRIC

# Load OpenAPI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Define constants and configurations 
DEVICE = "cpu" 
LR = 3e-4  
HIDDEN_SIZE = 64  
ENTROPY_LAMBDA = 0.05  
BASELINE_BETA = 0.9 
N_EPISODES = 50  
TEMPERATURE_LLM = 0.3  
MODEL_NAME_LLM = "gpt-4o-mini"  
EVAL_RETRIES = 3  
SAVE_INTERVAL = 10  
LOG_INTERVAL = 5 

# Exploration Parameters
EPSILON_START = 0.3  
EPSILON_END = 0.05   
EPSILON_DECAY = 0.995  

# Reward scaling parameters
REWARD_SCALE_FACTOR = 0.8  
REWARD_NOISE_STD = 2.0     
USE_COMPARATIVE_GRADING = True  

# Parse bins from the imported strings
def parse_bins(bin_string: str) -> List[str]:
    lines = bin_string.strip().split('\n')
    bins = []
    for line in lines:
        line = line.strip()
        if line:  
            clean_line = line.lstrip('- ').strip()
            bins.append(clean_line)
    return bins

# Parse the actual bins
CONCEPT_BIN_LIST = parse_bins(CONCEPT_BINS)
CONTEXT_BIN_LIST = parse_bins(CONTEXT_BINS)

# Create mappings
BIN2IDX_CONCEPT = {b: i for i, b in enumerate(CONCEPT_BIN_LIST)}
BIN2IDX_CONTEXT = {b: i for i, b in enumerate(CONTEXT_BIN_LIST)}

# Get keys and sizes for block options
BLOCK_KEYS = list(BLOCK_OPTIONS.keys())
BLOCK_SIZES = [len(BLOCK_OPTIONS[k]) for k in BLOCK_KEYS]

# Helper Functions
def one_hot(index: int, size: int) -> torch.Tensor:
    v = torch.zeros(size, device=DEVICE)
    v[index] = 1.0
    return v

def build_state_tensor(concept_bin: str, context_bin: str, terms_count: int = 0, citations_count: int = 0) -> torch.Tensor:
    c_vec = one_hot(BIN2IDX_CONCEPT.get(concept_bin, 0), len(CONCEPT_BIN_LIST))
    ctx_vec = one_hot(BIN2IDX_CONTEXT.get(context_bin, 0), len(CONTEXT_BIN_LIST))
    
    # Add document features
    doc_features = torch.tensor([
        terms_count,     # Number of terms extracted
        citations_count  # Number of citations extracted
    ], device=DEVICE, dtype=torch.float32)
    
    return torch.cat([c_vec, ctx_vec, doc_features])

def read_markdown_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def bin_document_concept(markdown_content: str) -> str:
    concept_list_str = "\n".join([f"- {bin_name}" for bin_name in CONCEPT_BIN_LIST])
    
    # Truncate content if too long for API
    content_preview = markdown_content[:2000] + "..." if len(markdown_content) > 2000 else markdown_content
    
    response = call_llm_with_retry(
        system_prompt="You are an expert in educational content analysis.",
        user_prompt=f"""Please categorize the following document content into one of these subject bins:
{concept_list_str}

Document content:
{content_preview}

Analyze the content and determine which academic subject it most closely relates to. Return only the bin name that best matches the document's subject matter.""",
        retries=2
    )
    
    if response:
        response_lower = response.lower()
        for bin_name in CONCEPT_BIN_LIST:
            if bin_name.lower() in response_lower:
                return bin_name
    
    return CONCEPT_BIN_LIST[-1] if CONCEPT_BIN_LIST else "Miscellaneous / Other"

def bin_document_context(markdown_content: str) -> str:
    context_list_str = "\n".join([f"- {bin_name}" for bin_name in CONTEXT_BIN_LIST])
    
    # Analyze document structure and style
    content_preview = markdown_content[:1500] + "..." if len(markdown_content) > 1500 else markdown_content
    
    response = call_llm_with_retry(
        system_prompt="You are an expert in document type classification.",
        user_prompt=f"""Please categorize the following document by its source type:
{context_list_str}

Document content:
{content_preview}

Look at the structure, formatting, style, and content to determine what type of educational material this is. Return only the bin name that best matches the document type.""",
        retries=2
    )
    
    if response:
        response_lower = response.lower()
        for bin_name in CONTEXT_BIN_LIST:
            if bin_name.lower() in response_lower:
                return bin_name
    
    return CONTEXT_BIN_LIST[-1] if CONTEXT_BIN_LIST else "Study group collaborative notes"

def extract_terms_and_citations(markdown_content: str) -> Tuple[List[Dict[str, str]], List[str]]:    
    extraction_prompt = f"""
Analyze the following document and extract:

1. KEY TERMS: Important concepts, terminology, or vocabulary with their definitions (if provided in the text)
2. CITATIONS: Important quotes, statistics, references, or notable statements

Document content:
{markdown_content}

Please respond in this exact JSON format:
{{
    "terms": [
        {{"term": "concept name", "definition": "definition from text or empty string if not provided"}},
        {{"term": "another concept", "definition": "its definition or empty string"}}
    ],
    "citations": [
        "Important quote or statistic from the text",
        "Another significant reference or statement"
    ]
}}

Extract ALL relevant terms and citations. Include technical terms, key concepts, important names, theories, formulas, etc. For definitions, use the exact text from the document when available.
"""
    
    response = call_llm_with_retry(
        system_prompt="You are an expert at extracting educational content from documents. Always respond with valid JSON.",
        user_prompt=extraction_prompt,
        retries=3
    )
    
    if not response:
        return [], []
    
    try:
        # Parse the JSON response
        data = json.loads(response)
        terms = data.get("terms", [])
        citations = data.get("citations", [])
        
        # Validate and clean the data
        cleaned_terms = []
        for term in terms:
            if isinstance(term, dict) and "term" in term and "definition" in term:
                cleaned_terms.append({
                    "term": str(term["term"]).strip(),
                    "definition": str(term["definition"]).strip()
                })
        
        cleaned_citations = []
        for citation in citations:
            if isinstance(citation, str) and citation.strip():
                cleaned_citations.append(citation.strip())
        
        print(f"Extracted {len(cleaned_terms)} terms and {len(cleaned_citations)} citations")
        return cleaned_terms, cleaned_citations
        
    except json.JSONDecodeError as e:
        print(f"Error parsing extraction response: {e}")
        print(f"Response was: {response[:200]}...")
        return [], []

def format_terms_for_prompt(terms: List[Dict[str, str]]) -> str:
    if not terms:
        return "No key terms extracted"
    
    formatted_terms = []
    for term in terms:
        if term["definition"]:
            formatted_terms.append(f"- {term['term']}: {term['definition']}")
        else:
            formatted_terms.append(f"- {term['term']}")
    
    return "\n".join(formatted_terms)

def format_citations_for_prompt(citations: List[str]) -> str:
    if not citations:
        return "No key citations extracted"
    
    formatted_citations = []
    for i, citation in enumerate(citations, 1):
        formatted_citations.append(f"{i}. {citation}")
    
    return "\n".join(formatted_citations)

def assemble_prompt(actions: Dict[str, int],
                    concept_bin: str,
                    context_bin: str,
                    terms: List[Dict[str, str]],
                    citations: List[str]) -> str:
    """Assemble the study guide generation prompt"""
    filled_blocks = {
        k: BLOCK_OPTIONS[k][idx] for k, idx in actions.items()
    }
    
    # Add all required template variables
    filled_blocks['concept_bin'] = concept_bin
    filled_blocks['context_bin'] = context_bin
    filled_blocks['terms'] = format_terms_for_prompt(terms)
    filled_blocks['citations'] = format_citations_for_prompt(citations)
    
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

def get_study_guide_response(prompt: str) -> str:
    return call_llm_with_retry(
        system_prompt="You are an expert educational content creator. Generate comprehensive study guides that include all relevant terms and citations from the source material.",
        user_prompt=prompt
    ) or "Failed to generate study guide."

def extract_score_from_feedback(feedback: str) -> float:
    try:
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', feedback)
        
        if numbers:
            score = float(numbers[-1])
            return min(max(score, 0), 100)
        
        return 50.00  
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 50.00  

def evaluate_response_strict(prompt: str, answer_text: str, terms: List[Dict[str, str]], citations: List[str], baseline_response: str = None) -> float:    
    # Format terms and citations for evaluation
    terms_text = format_terms_for_prompt(terms)
    citations_text = format_citations_for_prompt(citations)
    
    strict_evaluation_prompt = f"""
You are an EXTREMELY STRICT educational content evaluator. Be highly critical and look for ANY flaws.

Evaluation Rubric:
{EVALUATION_METRIC}

Original Prompt:
{prompt}

Expected Terms to Include:
{terms_text}

Expected Citations to Include:
{citations_text}

Generated Study Guide Response:
{answer_text}

STRICT EVALUATION INSTRUCTIONS:
1. Check if ALL extracted terms are adequately covered in the response
2. Verify that relevant citations are incorporated appropriately
3. Evaluate accuracy, completeness, citation quality, relevance, and clarity
4. Be very critical - a score of 100 should be EXTREMELY rare
5. Deduct heavily for:
   - Missing key terms from the extracted list
   - Failure to incorporate important citations
   - Inaccurate definitions or explanations
   - Poor organization or unclear presentation
6. A "good" response should score 60-75
7. An "excellent" response should score 76-85
8. Only truly exceptional responses deserve 86+

{f'BASELINE COMPARISON: Compare this response to this baseline: {baseline_response}' if baseline_response else ''}

Provide a detailed critique focusing on term coverage and citation usage, then give a STRICT numerical score between 0-100. Remember: be harsh!
"""
    
    grader_feedback = call_llm_with_retry(
        system_prompt="You are an extremely strict educational content evaluator who focuses on completeness and accuracy.",
        user_prompt=strict_evaluation_prompt
    )
    
    if not grader_feedback:
        return 50.0
    
    score = extract_score_from_feedback(grader_feedback)
    
    # Apply reward scaling and noise
    scaled_score = score * REWARD_SCALE_FACTOR
    noise = random.gauss(0, REWARD_NOISE_STD)
    final_score = max(0, min(100, scaled_score + noise))
    
    return final_score

def get_baseline_response(prompt: str) -> str:
    """Get a simple baseline response for comparison"""
    simple_prompt = f"Create a basic study guide for the content mentioned in this prompt: {prompt[:300]}..."
    return call_llm_with_retry(
        system_prompt="You are a basic study guide creator. Provide simple, brief explanations.",
        user_prompt=simple_prompt
    ) or "Basic study guide."

# Policy Network
class DocumentPromptPolicy(nn.Module):    
    def __init__(self):
        super().__init__()
        # Input: concept bins + context bins + document features
        input_dim = len(CONCEPT_BIN_LIST) + len(CONTEXT_BIN_LIST) + 2  # +2 for terms_count, citations_count
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
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
                if not deterministic and random.random() < epsilon:
                    action_idx = torch.randint(0, len(logit), (1,)).item()
                    dist = Categorical(logits=logit)
                else:
                    dist = Categorical(logits=logit)
                    if deterministic:
                        action_idx = torch.argmax(logit)
                    else:
                        action_idx = dist.sample()
                
                actions[key] = int(action_idx)
                if isinstance(action_idx, torch.Tensor):
                    log_probs.append(dist.log_prob(action_idx))
                else:
                    log_probs.append(dist.log_prob(torch.tensor(action_idx, dtype=torch.long, device=DEVICE)))
                entropies.append(dist.entropy())
        
        return actions, log_probs, entropies

# Data Classes 
@dataclass
class DocumentEpisodeLog:
    episode: int
    reward: float
    raw_reward: float  
    actions: Dict[str, int]
    prompt: str
    response: str
    concept_bin: str
    context_bin: str
    terms_count: int
    citations_count: int
    epsilon: float
    
    def to_dict(self):
        return asdict(self)

def save_episode_log(log: DocumentEpisodeLog, filename: str):
    with open(filename, "a") as f:
        f.write(json.dumps(log.to_dict()) + "\n")

def run_episode(policy: DocumentPromptPolicy,
                optimizer: torch.optim.Optimizer,
                baseline: float,
                concept_bin: str,
                context_bin: str,
                terms: List[Dict[str, str]],
                citations: List[str],
                episode: int,
                baseline_responses: Dict[str, str] = None) -> Tuple[float, DocumentEpisodeLog]:
    policy.train()
    
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
    
    # Build state tensor
    state_tensor = build_state_tensor(concept_bin, context_bin, len(terms), len(citations))
    
    # Get logits from policy network
    logits = policy(state_tensor)
    
    # Sample actions
    actions = {}
    log_probs = []
    entropies = []
    
    for key, logit in zip(BLOCK_KEYS, logits):
        if random.random() < epsilon:
            action_idx = random.randint(0, len(BLOCK_OPTIONS[key]) - 1)
            action_tensor = torch.tensor(action_idx, dtype=torch.long, device=DEVICE)
        else:
            dist = Categorical(logits=logit)
            action_tensor = dist.sample()
            action_idx = action_tensor.item()
        
        actions[key] = action_idx
        dist = Categorical(logits=logit)
        log_probs.append(dist.log_prob(action_tensor))
        entropies.append(dist.entropy())
    
    # Assemble prompt and get response
    prompt = assemble_prompt(actions, concept_bin, context_bin, terms, citations)
    response = get_study_guide_response(prompt)
    
    # Get baseline for comparison
    baseline_response = None
    if USE_COMPARATIVE_GRADING:
        if baseline_responses is None:
            baseline_responses = {}
        cache_key = f"{concept_bin}_{context_bin}_{len(terms)}_{len(citations)}"
        if cache_key not in baseline_responses:
            baseline_responses[cache_key] = get_baseline_response(prompt)
        baseline_response = baseline_responses[cache_key]
    
    # Evaluate with terms and citations
    raw_reward = evaluate_response_strict(prompt, response, terms, citations, baseline_response)
    reward = raw_reward
    
    # Update policy
    advantage = reward - baseline
    policy_loss = -(advantage) * torch.stack(log_probs).sum()
    entropy_loss = -ENTROPY_LAMBDA * torch.stack(entropies).sum()
    total_loss = policy_loss + entropy_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Create log
    log = DocumentEpisodeLog(
        episode=episode,
        reward=reward,
        raw_reward=raw_reward,
        actions=actions,
        prompt=prompt,
        response=response,
        concept_bin=concept_bin,
        context_bin=context_bin,
        terms_count=len(terms),
        citations_count=len(citations),
        epsilon=epsilon,
    )
    
    return reward, log

def print_epoch_summary(episode: int, reward: float, recent_rewards: List[float], best_reward: float, epsilon: float, logs: List[DocumentEpisodeLog]):
    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    recent_raw_rewards = [log.raw_reward for log in logs[-min(10, len(logs)):]]
    avg_raw_reward = sum(recent_raw_rewards) / len(recent_raw_rewards) if recent_raw_rewards else 0
    
    print(f"Episode {episode}/{N_EPISODES} | Reward: {reward:.2f} | Raw: {logs[-1].raw_reward:.2f} | "
          f"Avg: {avg_reward:.2f} | Raw Avg: {avg_raw_reward:.2f} | Best: {best_reward:.2f} | ε: {epsilon:.3f}")

def train_document_policy(markdown_file_path: str, filename: str = "document_training_logs.jsonl"):
    torch.manual_seed(42)
    random.seed(42)
    
    # Read and process the markdown file
    print("Reading and analyzing document...")
    markdown_content = read_markdown_file(markdown_file_path)
    if not markdown_content:
        print("Error: Could not read markdown file!")
        return None, []
    
    # Bin the document
    print("Binning document by subject and context...")
    concept_bin = bin_document_concept(markdown_content)
    context_bin = bin_document_context(markdown_content)
    
    # Extract terms and citations
    print("Extracting terms and citations...")
    terms, citations = extract_terms_and_citations(markdown_content)
    
    print(f"Document Analysis Complete:")
    print(f"- Subject: {concept_bin}")
    print(f"- Context: {context_bin}")
    print(f"- Terms extracted: {len(terms)}")
    print(f"- Citations extracted: {len(citations)}")
    print("="*60)
    
    # Initialize training
    policy = DocumentPromptPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    baseline = 0.0
    logs = []
    recent_rewards = []
    best_reward = float('-inf')
    best_episode = 0
    baseline_responses = {}
    
    # Clear log file
    if os.path.exists(filename):
        os.remove(filename)
    
    print(f"Starting training with {N_EPISODES} episodes...")
    print(f"Real-time logging to: {filename}")
    print("="*60)
    
    for episode in range(1, N_EPISODES + 1):
        reward, log = run_episode(
            policy=policy,
            optimizer=optimizer,
            baseline=baseline,
            concept_bin=concept_bin,
            context_bin=context_bin,
            terms=terms,
            citations=citations,
            episode=episode,
            baseline_responses=baseline_responses
        )
        
        baseline = BASELINE_BETA * baseline + (1 - BASELINE_BETA) * reward
        
        logs.append(log)
        recent_rewards.append(reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        
        if reward > best_reward:
            best_reward = reward
            best_episode = episode
        
        save_episode_log(log, filename)
        
        if episode % LOG_INTERVAL == 0:
            print_epoch_summary(episode, reward, recent_rewards, best_reward, log.epsilon, logs)
    
    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.2f} at episode {best_episode}")
    print(f"Final exploration rate: {logs[-1].epsilon:.3f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print(f"Document binned to: {concept_bin} | {context_bin}")
    
    actions, _, _ = policy.get_actions(
        build_state_tensor(concept_bin, context_bin, len(terms), len(citations)),
        deterministic=True
    )
    
    print("\nBest prompt configuration:")
    for key, idx in actions.items():
        print(f"{key}: {BLOCK_OPTIONS[key][idx]}")
    
    best_prompt = assemble_prompt(actions, concept_bin, context_bin, terms, citations)
    print("\nExample of best prompt template:")
    print(best_prompt[:500] + "..." if len(best_prompt) > 500 else best_prompt)
    
    return policy, logs

def main():
    # Test with a sample file - replace with actual file path
    # Default Values
    markdown_file = "samplefile.md"
    train_document_policy(markdown_file, filename="document_training_logs.jsonl")

if __name__ == "__main__":
    main()