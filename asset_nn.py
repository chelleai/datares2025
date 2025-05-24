import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import time

load_dotenv()
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-2.0-flash')

ENCODING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


BLOCK_OPTIONS = {
    "definition_style": [
        "concise",
        "technically accurate",
        "simple and easily understandable"
    ],
    "language_complexity": [
        "Use clear and plain language, avoiding overly complex terms.",
        "Use formal academic language.",
        "Match the complexity of the citations and use similar language.",
        "Make it understandable without oversimplifying technical terms."
    ],
    "relevant_terms": [
        "List 1-2 related terms, giving a simple 1 sentence description of the relationship.",
        "List 2-3 related terms, with no definition.",
        "Closely related concepts.",
        "Do not include any additional terms or concepts."
    ],
    "example_instructions": [
        "A real-world example of the term.",
        "Two contrasting examples of the concept.",
        "A scenario-based example that showcases the term."
    ],
    "analogy_style": [
        "A simple metaphor to explain the term.",
        "An analogy to a familiar and basic idea.",
        "Relate the term to an everyday object or concept."
    ],
    "practice_question": [
        "Construct one multiple choice question for basic recall.",
        "A True/False question testing recall of the definition.",
        "A True/False question testing an application of the term.",
        "A short answer question applying the concept."
    ]
}

PROMPT_TEMPLATE = """
You are generating content for a study guide. You are given:
    - A term: {term}
    - A list of cited quotations from a student-uploaded file that describes or defines the term: {citations}

Write a {definition_style} definition of the term, synthesizing information from the given citations.
{language_complexity}

Enhance the definition by including:
    - {relevant_terms}
    - {example_instructions}
    - {analogy_style}
    - {practice_question}
"""

EVALUATION_METRIC = """
Evaluation Metric for Asset Processing Pipeline

1. Accuracy (0.20)
Description: Are the extracted terms and the content that was generated, accurate, relevant, and properly cited? 
Score 1: Major inaccuracies; >50%% of extracted terms were wrong and not supported by citations
Score 2: Several minor inaccuracies or one big mistake (e.g., missing a key term).
Score 3: One minor error, but mostly accurate.
Score 4: Fully accurate with only a minor nitpicky issue.
Score 5: No errors or omissions, well cited and correct
Why it matters: Ensures that users have correct and reliable content

2. Completeness (0.20)
Description: Does the output capture all relevant terms from the content without omission?
Score 1: <40%% of relevant terms are extracted.
Score 2: 40-60%% of relevant terms are extracted. Misses critical concepts.
Score 3: Captures 61-80%% of terms. Some secondary omissions.
Score 4: Captures 81-99%% of terms. Minor omissions of less essential terms.
Score 5: 100%% of relevant terms are extracted.
Why it matters: Completeness ensures no significant information is left out, which is crucial for comprehensive analysis.

3. Citation Quality (0.15)
Description: Are the citations for each term relevant and accurate?
Score 1: Citations are mostly irrelevant or incorrect.
Score 2: Some citations are accurate, but there are major mismatches.
Score 3: Most citations are accurate, with minor inconsistencies.
Score 4: Citations are accurate and relevant, with one minor issue.
Score 5: All citations are correctly aligned with their respective terms.
Why it matters: Proper citation quality maintains the validity of the extracted terms and their sources.

4. Relevance (0.10)
Description: Are only meaningful and contextually appropriate terms extracted?
Score 1: Many extracted terms are irrelevant or unrelated (e.g., generic or off-topic).
Score 2: Several irrelevant terms are included.
Score 3: Mostly relevant, with a few unnecessary terms.
Score 4: One or two slightly off-topic terms, but overall appropriate.
Score 5: All terms are contextually relevant.
Why it matters: Extracting irrelevant terms can dilute the quality and focus of the processed content.

5. Clarity (0.10)
Description: Is the output formatted in a clear and understandable way?
Score 1: Output is disorganized, unstructured or difficult to follow.
Score 2: Several formatting or naming issues, poor readability.
Score 3: Mostly clear, but with 1-2 layout or structural inconsistencies.
Score 4: Well-organized and clear formatting, with only a minor formatting concern.
Score 5: Clear structure, easy to read, and logically structured.
Why it matters: Clarity ensures that users can easily understand the results, improving usability.

6. Robustness (0.10)
Description: Can the method handle a variety of content formats and structures?
Score 1: Fails with varied or complex content. Produces errors, incomplete outputs, or nonsensical results when faced with nonstandard formats.
Score 2: Works with simple content, but fails with moderate complexity.
Score 3: Handles most standard cases but struggles with complex ones.
Score 4: Handles diverse content, with one edge case issue.
Score 5: Robust across all tested content formats.
Why it matters: Robustness ensures the method works reliably across different input structures.

7. Efficiency (0.05)
Description: How fast and resource-efficient is the term extraction process?
Score 1: Very slow or computationally heavy.
Score 2: Slow for large files.
Score 3: Reasonably fast, with minor inefficiencies.
Score 4: Efficient, with only slight room for optimization.
Score 5: Highly efficient, fast processing even with large inputs.
Why it matters: Efficiency is crucial for processing large datasets or real-time applications.

8. Handling of Edge Cases (0.05)
Description: Does the method account for atypical or complex input formats?
Score 1: Fails completely on edge cases.
Score 2: Handles basic cases but fails on less common ones.
Score 3: Handles common edge cases, with one or two failures.
Score 4: Successfully handles most edge cases.
Score 5: Robustly handles all edge cases tested.
Why it matters: Edge cases are common in real-world data, so the system must be resilient.

9. Adaptability to New Content (0.05)
Description: How easily can the extraction method adapt to new types of input or updated formats?
Score 1: Completely rigid and unadaptable.
Score 2: Minor tweaks are possible but difficult.
Score 3: Some adaptability with manual adjustments.
Score 4: Mostly adaptable with a few modifications.
Score 5: Fully adaptable with minimal effort.
Why it matters: Adaptability ensures that the method remains useful as content formats evolve.

Weight Distribution:
Accuracy: 0.20
Completeness: 0.20
Citation Quality: 0.15
Relevance: 0.10
Clarity: 0.10
Robustness: 0.10
Efficiency: 0.05
Handling of Edge Cases: 0.05
Adaptability to New Content: 0.05
How to use these metrics: each metric is scored from 1 to 5 using the above criteria, multiply each score by its weight and sum for an evaluation score out of 5. Multiply this by 20 to obtain a final evaluation score out of 100 points.

"""




def encode_inputs(term: str, citations: list[str]) -> torch.Tensor:
    citations_text = " ".join(citations)
    text = f"Term: {term}. Citations: {citations_text}"
    embedding = ENCODING_MODEL.encode(text, convert_to_tensor=True)
    return embedding

def construct_prompt(term: str, citations: list[str], chosen_blocks: dict[str, int]) -> str:
    # fill template with blocks
    prompt = PROMPT_TEMPLATE.format(
        term = term,
        citations = " ".join(citations),
        definition_style = BLOCK_OPTIONS["definition_style"][chosen_blocks["definition_style"]],
        language_complexity = BLOCK_OPTIONS["language_complexity"][chosen_blocks["language_complexity"]],
        relevant_terms = BLOCK_OPTIONS["relevant_terms"][chosen_blocks["relevant_terms"]],
        example_instructions = BLOCK_OPTIONS["example_instructions"][chosen_blocks["example_instructions"]],
        analogy_style = BLOCK_OPTIONS["analogy_style"][chosen_blocks["analogy_style"]],
        practice_question = BLOCK_OPTIONS["practice_question"][chosen_blocks["practice_question"]]
    )
    return prompt

statement = ""


def add_numbers(a,b):
    return a + b

def add_numbers_hardcoded():
    return 8

def generate_text(prompt: str) -> str:
    try:
        response = MODEL.generate_content(prompt)
        time.sleep(4)  # Add 4 second delay between calls to stay under 15 requests per minute
        return response.text
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def evaluate_response(response: str) -> str:
    try:
        prompt = f"{EVALUATION_METRIC}\n\nAssistant Answer:\n{response}\n\nBe harsh when grading each score. Please provide a final score between 0 and 100 as a single number."
        evaluation = MODEL.generate_content(prompt)
        time.sleep(4)  # Add 4 second delay between calls
        return evaluation.text
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return ""


def extract_score(evaluation_text: str) -> float:
    score_lines = [
        line.lower() for line in evaluation_text.split("\n")
        if any(keyword in line.lower() for keyword in ["socre", "final", "result"])
    ] # extract lines with any indication of final score

    numbers = []
    for line in score_lines:
        numbers.extend(re.findall(r'\d+\.?\d*', line))

    if not numbers: # search all text if no numbers found
        numbers = re.findall(r'\d+\.?\d*', evaluation_text)
    
    if numbers:
        score = float(numbers[-1]) # last number is likely the score
        return min(max(score, 0), 100)
    else:
        return 50.0 # default to 50 if no score found
    

def save_logs(logs: List[Dict[str, Any]], filename: str = "training_logs.jsonl") -> None:
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    print(f"Logs saved to {filename}")



class PromptPolicy(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()

        # first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        # output layers for each parameter
        self.definition_style = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["definition_style"]))
        self.language_complexity = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["language_complexity"]))
        self.relevant_terms = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["relevant_terms"]))
        self.example_instructions = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["example_instructions"]))
        self.analogy_style = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["analogy_style"]))
        self.practice_question = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["practice_question"]))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        
        return {
            "definition_style": self.definition_style(x),
            "language_complexity": self.language_complexity(x),
            "relevant_terms": self.relevant_terms(x),
            "example_instructions": self.example_instructions(x),
            "analogy_style": self.analogy_style(x),
            "practice_question": self.practice_question(x)
        }




def train_step(policy: PromptPolicy, optimizer: torch.optim.Optimizer, term: str, citations: list[str]) -> Tuple[float, float, Dict[str, int], str, str, str]:
    embedding = encode_inputs(term, citations)
    embedding = embedding.to(DEVICE)

    policy = policy.to(DEVICE)
    policy.train()
    
    batch_size = 32
    batch_embedding = embedding.unsqueeze(0).repeat(batch_size, 1)
    
    output = policy(batch_embedding)
    output = {k: v[0:1] for k, v in output.items()}

    log_probs: List[torch.Tensor] = []
    actions: Dict[str, int] = {}

    for key, logits in output.items():
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        actions[key] = int(action.item())  # Convert to int explicitly
        log_probs.append(log_prob)

    prompt = construct_prompt(term, citations, actions)
    response = generate_text(prompt)
    evaluation = evaluate_response(response)
    reward = extract_score(evaluation)

    # compute loss
    total_log_prob = torch.stack(log_probs).sum()
    loss = -total_log_prob * reward

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), reward, actions, prompt, response, evaluation



def main():
    policy = PromptPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    term = "protein"
    citations = [
        "Each chromosome also includes proteins called histones that help to organize the DNA.",
        "Proteins influence traits The cell uses the information found in genes to build other molecules, primarily proteins.",
        "Proteins are molecules that perform many different functions in a cell.",
        "For example, proteins are involved in defense, storage, transport, cellular communication, movement, and maintaining cell structure.",
        "Proteins are made up of monomers called amino acids.",
        "Each protein is made up of one or more polypeptides, which are folded and coiled into a specific three-dimensional (3D) structure.",
        "This 3D structure determines the protein's function.",
        "So, how does the cell turn the information in genes into proteins?",
        "Specifically, the order of nucleotides in a gene determines the order of amino acids in one or more proteins.",
        "This means that variation in the order of nucleotides in a gene can produce variation in the order of amino acids in a protein, which can affect the protein's function.",
        "A protein is shown as a globular, or lumpy ball-like, shape.",
        "Stretching out from the protein is a string of amino acids, which are represented as circles.",
        "A protein is made up of subunits called amino acids.",
        "The order of nucleotides in a gene determines the order of amino acids in one or more proteins.",
        "An organism has many different genes, and so can produce many different proteins.",
        "These proteins carry out a variety of functions that, in turn, affect the organism's traits.",
        "In these cats, the coat color trait is influenced by the MC1R gene, which encodes the MC1R protein.",
        "The MC1R protein is a receptor located on the surface of cells responsible for producing melanin-the pigment that gives color to animals' skin and hair.",
        "Leopards and black panthers have different variations in the MC1R gene, and therefore different versions of the MC1R protein.",
        "In black panthers, the MC1R protein is highly active, resulting in a higher production of eumelanin and a darker coat.",
        "In leopards, the MC1R protein is less active, resulting in the golden coat color trait.",
        "A diagram shows the relationships between the M C 1 R gene, the M C 1 R protein, and the coat color trait.",
        "An arrow points from the gene to a structure embedded in a cell membrane labeled M C 1 R protein.",
        "An arrow points from the protein to two close-up images of coat patterns.",
        "The MC1R gene encodes the MC1R protein.",
        "The activity of the MC1R protein determines if a leopard will have a spotted or solid black coat.",
        "In summary, an organism's genes determine the structure and function of its proteins.",
        "These proteins in turn carry out functions in the cell that influence an organism's traits."
    ]
    
    logs = []

    for episode in range(1, 51):
        loss, reward, actions, prompt, response, evaluation = train_step(policy, optimizer, term, citations)
        print(f"Episode {episode} - Loss: {loss:.4f}, Reward: {reward:.2f}")

        # log each episode
        logs.append({
            "episode": episode,
            "reward": reward,
            "actions": actions,
            "prompt": prompt,
            "response": response,
            "evaluation": evaluation
        })

        # save logs periodically
        if episode % 10 == 0:
            save_logs(logs)
            print(f"Episode {episode} — Loss: {loss:.4f}, Reward: {reward:.2f}, Actions: {actions}")

    save_logs(logs)





if __name__ == "__main__":
    main()

