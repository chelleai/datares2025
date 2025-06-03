import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import time
from markdown_it import MarkdownIt

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = genai.GenerativeModel('gemini-2.0-flash')
DEVICE = "cpu"
EVALUATION_METRIC = """# Evaluation Metric for Grading Study Guides

### Accuracy (4)
Are the extracted terms and generated content factually accurate?
- 1: Multiple major inaccuracies or misinterpretations.
- 2: One clear mistake or multiple nitpicky issues.
- 3: Mostly accurate with minor errors.
- 4: No obvious errors, very solid.
- 5: Perfect factual precision.

### Completeness (4)
Are all required elements included (definitions, examples, analogies, summary, etc.)?
- 1: Missing many components.
- 2: Several missing parts.
- 3: One or two missing or weak sections.
- 4: All parts are present and well-formed.
- 5: Fully complete and well-integrated.

### Relevance (3)
Are examples and analogies contextually appropriate?
- 1: Off-topic or distracting content.
- 3: Mostly relevant with small distractions.
- 5: Fully relevant and on-topic.

### Contextual and Subject Alignment (4)
Is the tone, format, and complexity suited to the subject/context?
- 1: Mismatch with the intended tone or audience.
- 3: Mostly appropriate.
- 5: Fully aligned in tone, depth, and style.

### Clarity (3)
Is the content well-structured and understandable?
- 1: Disorganized or hard to follow.
- 3: Mostly clear.
- 5: Very clear and structured.

### Conciseness and Depth (2)
Is the explanation brief yet insightful?
- 1: Too verbose or shallow.
- 2: Adequate.
- 5: Perfect balance.

## Scoring
Multiply each category by its weight. Total score is out of 100."""

MARKDOWN_FILE = """Chapter I - The Old Sea-dog at the 'Admiral Benbow'
Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having asked me to write down the whole particulars about Treasure Island, from the beginning to the end...
I remember him as if it were yesterday, as he came plodding to the inn door, his sea-chest following behind him in a hand-barrow — a tall, strong, heavy, nut-brown man, his tarry pigtail falling over the shoulder of his soiled blue coat...
“Fifteen men on the dead man’s chest — Yo-ho-ho, and a bottle of rum!” in the high, old tottering voice...
There were nights when he took a deal more rum and water than his head would carry; and then he would sometimes sit and sing his wicked, old, wild sea-songs...
“If you do not put that knife this instant in your pocket, I promise, upon my honour, you shall hang at the next assizes.”
And so the mystery of the one-legged seafaring man haunted my dreams..."""


BLOCK_OPTIONS = {
    "definition_style": ["concise", "technically accurate", "simple and easily understandable", "formal and objective"],
    "language_complexity": [
        "Use clear and plain language, avoiding overly complex terms.",
        "Use formal academic language.",
        "Match the complexity of the citations and use similar language.",
        "Make it understandable without oversimplifying technical terms."
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
    "summary": [
        "A brief, 2-3 sentence summary of core themes across terms and citations.",
        "A detailed, 4-5 paragraph connecting key concepts.",
        "A single-sentence abstract capturing the most significant insight or take away from the content.",
        "A 3-5 bullet-point list highlighting the main findings or conclusions."
    ],
    "key_topics": [
        "List 3-5 recurring themes based on citation frequency.",
        "Identify 2-3 foundational concepts critical for subject mastery.",
        "Identify 2-4 core terms or jargon essential to understanding the material."
    ],
    "relevant_terms": [
        "List 3-4 related terms, giving a simple 1 sentence description of how they are related to the text.",
        "List 4-5 related terms, with no definition.",
        "1-2 closely related concepts."
    ],
    "practice_question": [
        "2-3 multiple choice questions testing basic recall.",
        "2-3 True/False questions testing recall of definitions or applications of the terms.",
        "2-3 short answer questions applying the concept.",
        "1 True/False, 1 multiple choice, and 1 short answer question to test recall of definitions and key concepts.",
        "1 True/False, 1 multiple choice, and 1 short answer question to test applications of key concepts."
    ]
}

PROMPT_TEMPLATE = """
You are generating content for a study guide. You are given a list of key terms and for each term, a list of citations providing context for the term.
Write a {definition_style} definition for each term, synthesizing information from the given citations. {language_complexity}
Enhance each definition by including:
    - {example_instructions}
    - {analogy_style}
For the entire document, provide:
    - {summary}
    - {key_topics}
    - {relevant_terms}
    - {practice_question}
Here is the provided context: {context}
Here is the provided list of key terms and citations: {terms_and_citations}
"""

CONTEXT_PROMPT = "Please give me the context of the following text.\n{text}"
TERM_CITATION_PROMPT = "Given this markdown file, extract the key terms...\n{text}"
CONTEXT_BIN_PROMPT = "You are given a list of context bins...\nHere is the list of context bins:{context_bins}\nHere is the list of terms and citations:{context}"
SUBJECT_BIN_PROMPT = "You are given a list of subject bins...\nHere is the list of subject bins:{subject_bins}\nHere is the text containing the terms and citations:{terms_citations}"

CONTEXT_BINS = ["Textbook", "Interview", "Song Lyrics", "News Article", "Blog Article", "Academic Paper", "Research Paper", "Dictionary", "Encyclopedia", "Documentary", "Fiction Book", "Nonfiction Book", "Conversation"]
SUBJECT_BINS = ["Math", "Statistics", "Calculus", "Linear Algebra", "Geometry", "Algebra", "Arithmetic", "Discrete Mathematics", "English", "Literature", "English Literature", "Grammar", "Poetry", "Journalism", "Shakespeare", "Science", "Biology", "Chemistry", "Astronomy", "Earth Science", "Physics", "Biochemistry", "Environmental Science", "Psychology", "Geography", "Geology", "Anatomy", "Sociology", "Economics", "Business", "Communications", "Political Science", "Government", "Management", "Accounting", "Finance", "Public Affairs", "Social Studies", "History", "American History", "European History", "World History", "Asian History", "African History", "Foreign Language", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Hindi", "Portuguese", "Italian", "Latin", "Technology", "Computer Science", "Data Science", "Architecture", "Engineering", "Software Engineering", "Chemical Engineering", "Bioengineering", "Electrical Engineering", "Industrial Engineering", "Mechanical Engineering", "Civil Engineering", "Aerospace Engineering", "Medicine", "Law", "Education", "Life Science", "Physical Science", "Social Science", "Humanities", "Art", "Design", "Marketing", "Performing Arts", "Philosophy", "Religion", "Linguistics", "Ethics", "Music", "Physical Education", "Health", "Cooking", "Fashion", "Pop Culture", "TV Shows", "Movies", "Celebrities", "Film", "Lifestyle", "Transportation", "Social Media", "Memes", "Video Games", "Games", "Sports", "Animals", "Food", "Holidays", "Gardening", "Hospitality", "Everyday Life", "Jobs"]

def preprocess_markdown(markdown: str) -> Dict[str, str]:
    paragraphs = [p.strip() for p in markdown.split("\n\n") if len(p.strip()) > 0]
    context_section = "\n\n".join(paragraphs[:2])
    quotes = re.findall(r"“([^”]+)”", markdown) + re.findall(r'"([^"]+)"', markdown)
    citations_section = "\n".join(quotes)
    cleaned_text = re.sub(r'\s+', ' ', markdown)
    md = MarkdownIt()
    tokens = md.parse(markdown)
    headings = [t.content for t in tokens if t.type == "heading_open"]
    return {"context": context_section, "citations": citations_section, "cleaned": cleaned_text, "headings": "\n".join(headings)}

def encode_bins(context_bin: str, subject_bin: str) -> torch.Tensor:
    c_idx = CONTEXT_BINS.index(context_bin)
    s_idx = SUBJECT_BINS.index(subject_bin)
    c_hot = torch.zeros(len(CONTEXT_BINS))
    s_hot = torch.zeros(len(SUBJECT_BINS))
    c_hot[c_idx] = 1
    s_hot[s_idx] = 1
    return torch.cat([c_hot, s_hot])

def construct_prompt(context: str, terms_citations: str, chosen_blocks: dict[str, int]) -> str:
    return PROMPT_TEMPLATE.format(
        context=context,
        terms_and_citations=terms_citations,
        **{k: BLOCK_OPTIONS[k][chosen_blocks[k]] for k in BLOCK_OPTIONS}
    )

def call_llm(prompt: str) -> str:
    try:
        response = MODEL.generate_content(contents=prompt, generation_config={"temperature": 0})
        time.sleep(4)
        return response.text
    except Exception as e:
        print(f"LLM error: {e}")
        return ""

def extract_score(eval_text: str) -> float:
    lines = [l.lower() for l in eval_text.split("\n") if "score" in l or "final" in l]
    nums = []
    for l in lines:
        nums.extend(re.findall(r'\d+\.?\d*', l))
    if not nums:
        nums = re.findall(r'\d+\.?\d*', eval_text)
    return min(max(float(nums[-1]), 0), 100) if nums else 50.0

def save_logs(logs: List[Dict[str, Any]], filename: str = "training_logs.jsonl") -> None:
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

class PromptPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        dim = len(CONTEXT_BINS) + len(SUBJECT_BINS)
        h = 128
        self.fc1 = nn.Linear(dim, h)
        self.fc2 = nn.Linear(h, h)
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({k: nn.Linear(h, len(v)) for k, v in BLOCK_OPTIONS.items()})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.drop(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return {k: head(x) for k, head in self.heads.items()}

def train_step(policy: PromptPolicy, optimizer: torch.optim.Optimizer, markdown: str) -> Tuple:
    parsed = preprocess_markdown(markdown)
    context = call_llm(CONTEXT_PROMPT.format(text=parsed["context"]))
    terms = call_llm(TERM_CITATION_PROMPT.format(text=parsed["citations"]))
    context_bin = call_llm(CONTEXT_BIN_PROMPT.format(context_bins=CONTEXT_BINS, context=context)).strip()
    subject_bin = call_llm(SUBJECT_BIN_PROMPT.format(subject_bins=SUBJECT_BINS, terms_citations=terms)).strip()
    x = encode_bins(context_bin, subject_bin).to(DEVICE)
    policy.train()
    out = policy(x)
    actions, logs = {}, []
    for k, logits in out.items():
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        a = dist.sample()
        actions[k] = int(a.item())
        logs.append(dist.log_prob(a))
    prompt = construct_prompt(context, terms, actions)
    resp = call_llm(prompt)
    evaluation = call_llm(f"Evaluate this: {resp}")
    reward = extract_score(evaluation)
    loss = -torch.stack(logs).sum() * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), reward, context_bin, subject_bin, actions, prompt, resp, evaluation

def create_study_guide(markdown: str) -> Tuple[Dict, Dict]:
    policy = PromptPolicy().to(DEVICE)
    opt = optim.Adam(policy.parameters(), lr=1e-4)
    best, worst = {"reward": -1}, {"reward": 101}
    logs, rewards = [], []
    for ep in range(1, 51):
        loss, reward, ctx, subj, actions, prompt, resp, eval = train_step(policy, opt, markdown)
        logs.append({"episode": ep, "reward": reward, "context": ctx, "subject": subj, "actions": actions})
        rewards.append(reward)
        if reward > best["reward"]: best.update({"reward": reward, "actions": actions, "prompt": prompt, "response": resp, "evaluation": eval})
        if reward < worst["reward"]: worst.update({"reward": reward, "actions": actions, "prompt": prompt, "response": resp, "evaluation": eval})
        if ep % 5 == 0: save_logs(logs)
        if ep >= 10 and reward > np.mean(rewards) + 2 * np.std(rewards): break
    save_logs(logs)
    return best, worst

if __name__ == "__main__":
    markdown = MARKDOWN_FILE
    best, worst = create_study_guide(markdown)
    print("--- BEST ---\n", best["response"])
    print("--- WORST ---\n", worst["response"])
