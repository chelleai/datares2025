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

load_dotenv()
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
MODEL = genai.GenerativeModel('gemini-2.0-flash')
DEVICE = "cpu"


BLOCK_OPTIONS = {
    "definition_style": [
        "concise",
        "technically accurate",
        "simple and easily understandable",
        "formal and objective"
    ],
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
        "Identify 2-3 foundational concepts critical for subject mastery."
        "Identify 2-4 core terms or jargon essential to understanding the material.",
    ],
    "relevant_terms": [
        "List 3-4 related terms, giving a simple 1 sentence description of how they are related to the text.",
        "List 4-5 related terms, with no definition.",
        "1-2 closely related concepts.",
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
You are generating content for a study guide.
You are given a list of key terms and for each term, a list of citations providing context for the term.

Write a {definition_style} definition for each term, synthesizing information from the given citations. {language_complexity}

Enhance each definition by including:
    - {example_instructions}
    - {analogy_style}

For the entire document, provide:
    - {summary}
    - {key_topics}
    - {relevant_terms}
    - {practice_question}

Here is the provided context:
{context}

Here is the provided list of key terms and citations:
{terms_and_citations}
"""



# prompts to extract information from the markdown file
CONTEXT_PROMPT = (
    "Given the following text, identify its main topic and the type of source (e.g., lecture, podcast, article, etc.). "
    "Be concise and specific. The text is:\n{text}"
)
TERM_CITATION_PROMPT = (
    "From this text, extract the most important terms for understanding the topic. For each term, list relevant quoted passages from the text. "
    "Clearly associate each citation with its term. Only output the terms and their citations. The text is:\n{text}"
)

# prompts to classify the terms+citations into bins
CONTEXT_BIN_PROMPT = (
    "You are provided with a list of context bins describing possible source types. "
    "Given the context below, select the single most appropriate bin from the list. "
    "Return only the bin name, with no extra text or whitespace.\n"
    "Context bins: {context_bins}\n"
    "Context: {context}"
)

SUBJECT_BIN_PROMPT = (
    "You are provided with a list of subject bins describing possible topics. "
    "Given the terms and citations below, select the most specific subject bin that fits all the content. "
    "Return only the bin name, with no extra text or whitespace.\n"
    "Subject bins: {subject_bins}\n"
    "Terms and citations: {terms_citations}"
)

# prompt for evulation
EVALUATION_PROMPT = (
    "{eval_metric}\n\n"
    "Context: {context}\n"
    "Subject: {subject}\n"
    "Study Guide:\n{response}\n\n"
    "Grade each criterion strictly. Give a final score between 0 and 100 as a single number at the end."
)

# bins
CONTEXT_BINS = [
    "Lecture", "Podcast", "Magazine Article", "Blog Post", 
    "Documentary", "Online Course", "Reference Book", 
    "Dataset", "White Paper", "Interview", "Presentation Slides"
]
SUBJECT_BINS = [
    "Math", "Statistics", "Calculus", "Linear Algebra", "Geometry", "Algebra", "Arithmetic", "Discrete Mathematics", "Probability", "Number Theory", "Topology", "Differential Equations", "Combinatorics", "Set Theory", "Mathematical Logic",
    "English", "Literature", "English Literature", "Grammar", "Poetry", "Journalism", "Shakespeare", "Linguistics", "Creative Writing", "World Literature", "Comparative Literature", "Rhetoric", "Classical Literature", "Children's Literature", "Literary Theory",
    "Science", "Biology", "Chemistry", "Astronomy", "Earth Science", "Physics", "Biochemistry", "Environmental Science", "Psychology", "Geography", "Geology", "Anatomy", "Sociology", "Botany", "Zoology", "Genetics", "Microbiology", "Ecology", "Neuroscience", "Marine Biology",
    "Economics", "Business", "Communications", "Political Science", "Government", "Management", "Accounting", "Finance", "Public Affairs", "Anthropology", "Criminology", "International Relations", "Sociology", "Human Geography", "Demography", "Public Policy", "Urban Studies", "Gender Studies", "Cultural Studies", "Archaeology",
    "Social Studies", "History", "American History", "European History", "World History", "Asian History", "African History", "Middle Eastern History", "Latin American History", "Ancient History", "Medieval History", "Modern History", "Military History", "Art History", "History of Science",
    "Foreign Language", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Hindi", "Portuguese", "Italian", "Latin", "Russian", "Arabic", "Greek", "Hebrew", "Turkish", "Dutch", "Swedish", "Polish", "Vietnamese",
    "Technology", "Computer Science", "Data Science", "Architecture", "Engineering", "Software Engineering", "Chemical Engineering", "Bioengineering", "Electrical Engineering", "Industrial Engineering", "Mechanical Engineering", "Civil Engineering", "Aerospace Engineering", "Robotics", "Information Technology", "Cybersecurity", "Artificial Intelligence", "Machine Learning", "Web Development", "Game Development",
    "Medicine", "Law", "Education", "Life Science", "Physical Science", "Social Science", "Humanities", "Nursing", "Pharmacy", "Dentistry", "Veterinary Science", "Public Health", "Nutrition", "Epidemiology", "Kinesiology", "Occupational Therapy", "Speech Pathology", "Genomics", "Pathology", "Radiology",
    "Art", "Design", "Marketing", "Performing Arts", "Philosophy", "Religion", "Ethics", "Music", "Film Studies", "Photography", "Sculpture", "Painting", "Theater", "Dance", "Fashion", "Interior Design", "Graphic Design", "Art Criticism", "Aesthetics", "Cultural Anthropology",
    "Physical Education", "Health", "Cooking", "Pop Culture", "TV Shows", "Movies", "Celebrities", "Film", "Lifestyle", "Transportation", "Social Media", "Memes", "Video Games", "Games", "Sports", "Animals", "Food", "Holidays", "Gardening", "Hospitality", "Everyday Life", "Jobs", "Entrepreneurship", "Blockchain", "Sustainability", "Climate Science", "Space Exploration", "Forensic Science", "Disaster Management", "Human Rights", "Peace Studies", "Aging Studies", "Bioethics", "Astrobiology", "Meteorology", "Oceanography", "Agriculture", "Forestry", "Wildlife Conservation", "Tourism", "Event Management", "Library Science", "Museum Studies", "Cartography", "Logistics", "Supply Chain", "Retail", "Customer Service", "Advertising", "E-Commerce", "Real Estate", "Insurance", "Banking", "Investment", "Actuarial Science", "Statistics Education", "Science Communication", "Media Studies", "Journalism Ethics", "Digital Humanities", "Cognitive Science", "Behavioral Science", "Speech Communication", "Debate", "Public Speaking"
]



# test markdown file
MARKDOWN_FILE = """
Chapter II - The Process of Photosynthesis
The process of photosynthesis is fundamental to life on Earth. It is the way plants, algae, and some bacteria convert light energy into chemical energy, which is stored in glucose molecules. This process is vital for the production of food in plants and for the production of oxygen, which is crucial for all aerobic life forms.

The chemical equation for photosynthesis can be summarized as follows:

6CO2 + 6H2O + light energy → C6H12O6 + 6O2
 
This reaction takes place in the chloroplasts of plant cells, where sunlight is absorbed by chlorophyll, the green pigment that captures light. This sunlight provides the energy needed to drive the conversion of carbon dioxide and water into glucose and oxygen.

The Stages of Photosynthesis
Photosynthesis consists of two major stages, each responsible for different parts of the process.

1. The Light-dependent Reactions

The first stage of photosynthesis, the light-dependent reactions, occurs in the thylakoid membranes of the chloroplasts. During this stage, chlorophyll absorbs light energy, which is used to split water molecules into oxygen, protons, and electrons. The electrons travel through an electron transport chain, generating ATP and NADPH, two energy-rich molecules that will be used in the second stage of photosynthesis.

Key points:

Light energy splits water molecules, releasing oxygen.
ATP and NADPH are produced for the next stage of photosynthesis.
2. The Calvin Cycle (Light-independent Reactions)

The second stage, the Calvin Cycle, takes place in the stroma of the chloroplast. Here, ATP and NADPH produced in the light-dependent reactions are used to fix carbon dioxide from the atmosphere into an organic molecule. The carbon atoms from CO2 are incorporated into glucose, a sugar that serves as an energy source for the plant.

Key points:

ATP and NADPH are used to convert carbon dioxide into glucose.
This cycle repeats, producing sugars needed for the plant’s growth and energy.
Factors Affecting Photosynthesis
Several factors influence the rate at which photosynthesis occurs. These include:

Light Intensity: The more light available, the faster the rate of photosynthesis, but only up to a certain point.
Carbon Dioxide Concentration: Increased carbon dioxide levels can boost the photosynthesis rate, as it is one of the key reactants.
Temperature: Photosynthesis is most efficient within a specific temperature range. Extreme temperatures can slow the process down or even stop it altogether.
Example:

High Light Intensity: Increases the amount of energy available for photosynthesis.
Low CO2 Levels: Restricts the amount of glucose that can be produced.
The Importance of Photosynthesis
Photosynthesis is essential for life on Earth. It provides the energy necessary for plant growth and contributes oxygen to the atmosphere, which is required for respiration in animals and humans.

Real-world applications:

Agriculture: Optimizing photosynthesis is key to increasing crop yields.
Climate Change: Plants’ role in removing CO2 from the atmosphere helps regulate the Earth’s climate.
Summary
Photosynthesis is a vital biochemical process that sustains plant life by converting light energy into chemical energy. It involves the light-dependent reactions and the Calvin Cycle, which together produce glucose and oxygen. This process is fundamental not only for plants but for the entire ecosystem, as it supports all life forms dependent on oxygen and energy.
"""



EVALUATION_METRIC = """
# Evaluation Metric for Grading Study Guides

### Accuracy (4)
**Description:** Are the extracted terms and content generated factually accurate?
**Score 1:** More than one major inaccurate statement or multiple minor inaccurate statements. 
**Score 2:** One minor nitpicky issue with one statement.
**Score 5** No errors or omissions, all statements are 100%% correct and there is not a single nitpick.
**Why it matters:** Ensures that users have correct and reliable content

### Completeness (4)
**Description:** Does the study guide include all the key features such as a summary, key topics, other relevant terms, and practice questions? Are there definitions, examples, and analogies for all key terms?
**Score 1:** Less than 50%% of the required features are present, and many terms lack definitions, examples, or analogies.
**Score 2:** At least one required feature or component is completely missing.
**Score 3:** One major feature is missing or incomplete. All key terms have definitions, examples, and analogies, but 1-2 omissions remain.
**Score 4:** All required features are present and complete. All key terms have definitions, examples, and analogies, with 1 minor omission or slight incompleteness.
**Score 5** Every required feature is fully present and complete. All key terms have comprehensive definitions, examples, and analogies, with no omissions.
**Why it matters:** Capturing all specified features makes sure all the study guide is more complete.

### Relevance (3)
**Description:** Are all definitions, examples, and analogies directly relevant to the topic discussed and contextually appropriate?
**Score 1:** More than 3 irrelevant or off-topic inclusions, especially those that clash with the overall topic of the text.
**Score 2:** 1-2 off-topic or tangential points.
**Score 3:** Mostly relevant, with 1-2 unnecessary elements that don't distract from the topic.
**Score 4:** 1-2 very minor irrelevant inclusions that don’t distract from the topic.
**Score 5:** All content is tightly focused and relevant for the context and subject.
**Why it matters:** Irrelevant information can dilute the quality and focus of the processed content.

### Contextual and Subject Alignment (4)
**Description:** Is the study guide tailored to the context and subject in tone, language complexity, example/analogy style, and practice question type?
**Score 1:** Output ignores context/subject; tone, complexity, or question type are inappropriate.
**Score 2:** 2-3 mismatches to context or subject expectations.
**Score 3:** Mostly appropriate, but 1-2 notable mismatches in tone, format, or question type.
**Score 4:** Well-matched with only a minor contextual slip.
**Score 5:** Fully tailored to both context and subject; matches expected tone, complexity, and question style throughout.
**Why it matters:** Different contexts can change how a study guide should present its tone and language complexity. Additionally, different study guide styles are suitable for different subjects. For example, math and science subjects would prefer more precise definitions and application-based questions, while history and language studies would require more recall and interpretation.

### Clarity (3)
**Description:** Is the content presented in a well-organized and understandable manner?
**Score 1:** Content is very disorganized or overly complex.
**Score 2:** Several formatting or naming issues, poor readability.
**Score 3:** Mostly clear, but with 1-2 layout or structural inconsistencies.
**Score 4:** Well-organized and clear formatting, with only one minor formatting concern.
**Score 5:** Clear structure; easy to read and logically structured.
**Why it matters:** Clarity ensures that users can easily understand the results, improving usability.

### Conciseness and Depth (2)
**Description:** Are explanations succinct, avoiding unnecessary verbosity, yet deep enough for subject mastery as appropriate for the context?
**Score 1:** Overly verbose or superficial; main ideas obscured, especially relative to context.
**Score 2:** Somewhat wordy or shallow; some unnecessary detail or lack of depth.
**Score 3:** Mostly concise, with minor tangents or shallow spots.
**Score 4:** Generally succinct, with appropriate depth for context/subject.
**Score 5:** Perfect balance of brevity and depth for the specific context and subject.
**Why it matters:** Long-winded explanations lose attention and overwhelm readers.

## Weight Distribution:
Accuracy: 4
Completeness: 4
Relevance: 3
Contextual and Subject Alignment: 4
Clarity: 3
Conciseness and Depth: 2

## How to use these metrics:
Score each metric from 1 to 5 using the above criteria. Keep in mind the provided context and subject when scoring, thinking about how they affect the structure of a study guide.
Multiply each score by its weight and sum to get a final score out of 100. You should include your final sum calculation in your evaluation, with the final score at the very end.
"""







def encode_bin(context_bin: str, subject_bin: str) -> torch.Tensor:
    context_index = CONTEXT_BINS.index(context_bin)
    subject_index = SUBJECT_BINS.index(subject_bin)

    # one hot encodings
    context_one_hot = torch.zeros(len(CONTEXT_BINS))
    subject_one_hot = torch.zeros(len(SUBJECT_BINS))
    context_one_hot[context_index] = 1
    subject_one_hot[subject_index] = 1

    return torch.cat([context_one_hot, subject_one_hot])


def construct_prompt(context: str, terms_citations: str, chosen_blocks: dict[str, int]) -> str:
    # fill template with blocks
    prompt = PROMPT_TEMPLATE.format(
        context = context,
        terms_and_citations = terms_citations,
        definition_style = BLOCK_OPTIONS["definition_style"][chosen_blocks["definition_style"]],
        language_complexity = BLOCK_OPTIONS["language_complexity"][chosen_blocks["language_complexity"]],
        example_instructions = BLOCK_OPTIONS["example_instructions"][chosen_blocks["example_instructions"]],
        analogy_style = BLOCK_OPTIONS["analogy_style"][chosen_blocks["analogy_style"]],
        summary = BLOCK_OPTIONS["summary"][chosen_blocks["summary"]],
        key_topics = BLOCK_OPTIONS["key_topics"][chosen_blocks["key_topics"]],
        relevant_terms = BLOCK_OPTIONS["relevant_terms"][chosen_blocks["relevant_terms"]],
        practice_question = BLOCK_OPTIONS["practice_question"][chosen_blocks["practice_question"]]
    )
    return prompt


def call_llm(prompt: str) -> str:
    try:
        response = MODEL.generate_content(
            contents=prompt,
            generation_config = {
                "temperature": 0,
            }
        )
        time.sleep(4) # add 4 second delay between calls to stay under 15 requests per minute
        return response.text
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""


def get_score(evaluation_text: str) -> float:
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


def save_logs(logs: List[Dict[str, Any]], filename: str = "training_logs_update.jsonl") -> None:
    with open(filename, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    print(f"Logs saved to {filename}")



class PromptPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = len(CONTEXT_BINS) + len(SUBJECT_BINS)
        hidden_dim = 128

        # first hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        # output layers for each parameter
        self.definition_style = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["definition_style"]))
        self.language_complexity = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["language_complexity"]))
        self.example_instructions = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["example_instructions"]))
        self.analogy_style = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["analogy_style"]))
        self.summary = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["summary"]))
        self.key_topics = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["key_topics"]))
        self.relevant_terms = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["relevant_terms"]))
        self.practice_question = nn.Linear(hidden_dim, len(BLOCK_OPTIONS["practice_question"]))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        
        return {
            "definition_style": self.definition_style(x),
            "language_complexity": self.language_complexity(x),
            "example_instructions": self.example_instructions(x),
            "analogy_style": self.analogy_style(x),
            "summary": self.summary(x),
            "key_topics": self.key_topics(x),
            "relevant_terms": self.relevant_terms(x),
            "practice_question": self.practice_question(x)
        }




def train_step(policy: PromptPolicy, optimizer: torch.optim.Optimizer, markdown_content: str) -> Tuple[float, float, str, str, Dict[str, int], str, str, str]:
    # extract context, terms, and citations from markdown
    context = call_llm(CONTEXT_PROMPT.format(text=markdown_content))
    terms_citations = call_llm(TERM_CITATION_PROMPT.format(text=markdown_content))

    # sort into bins
    context_bin = call_llm(CONTEXT_BIN_PROMPT.format(context_bins=CONTEXT_BINS, context=context)).strip()
    subject_bin = call_llm(SUBJECT_BIN_PROMPT.format(subject_bins=SUBJECT_BINS, terms_citations=terms_citations)).strip()
    
    encoding = encode_bin(context_bin, subject_bin).to(DEVICE)

    policy = policy.to(DEVICE)
    policy.train()
    
    output = policy(encoding)
    output = {k: v.squeeze() for k, v in output.items()}

    log_probs: List[torch.Tensor] = []
    actions: Dict[str, int] = {}

    for key, logits in output.items():
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        actions[key] = int(action.item())  # Convert to int explicitly
        log_probs.append(log_prob)

    prompt = construct_prompt(context, terms_citations, actions)
    response = call_llm(prompt)
    evaluation = call_llm(EVALUATION_PROMPT.format(eval_metric=EVALUATION_METRIC, context=context_bin, subject=subject_bin, response=response))
    reward = get_score(evaluation)

    # compute loss
    total_log_prob = torch.stack(log_probs).sum()
    loss = -total_log_prob * reward

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), reward, context_bin, subject_bin, actions, prompt, response, evaluation



def create_study_guide(markdown: str = MARKDOWN_FILE) -> Tuple[Dict[str, object], Dict[str, object]]:
    policy = PromptPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    logs = []
    best: Dict[str, object] = {
        "reward":  -float("inf"),
        "actions": None,
        "prompt": None,
        "response": None,
        "evaluation": None
    }
    worst: Dict[str, object] = {
        "reward": float("inf"),
        "actions": None,
        "prompt": None,
        "response": None,
        "evaluation": None
    }

    reward_history = []
    max_episodes = 50
    min_episodes = 10
    patience = 5
    no_improve_count = 0
    best_reward = -float("inf")

    for episode in range(1, max_episodes + 1):
        loss, reward, context, subject, actions, prompt, response, evaluation = train_step(policy, optimizer, markdown)
        reward_history.append(float(reward))

        print(f"Episode {episode} - Loss: {loss:.4f}, Reward: {reward:.2f}")

        # Track best reward
        if reward > best_reward:
            best_reward = reward
            no_improve_count = 0
            best.update({
                "reward": reward,
                "actions": actions,
                "prompt": prompt,
                "response": response,
                "evaluation": evaluation
            })
            print(f"New best reward: {best_reward:.2f}")
        else:
            no_improve_count += 1

        # Track worst reward
        if reward < worst["reward"]:
            worst.update({
                "reward": reward,
                "actions": actions,
                "prompt": prompt,
                "response": response,
                "evaluation": evaluation
            })

        logs.append({
            "episode": episode,
            "reward": reward,
            "context": context,
            "subject": subject,
            "actions": actions,
            "prompt": prompt,
            "response": response,
            "evaluation": evaluation
        })

        # Save logs periodically
        if episode % 5 == 0:
            save_logs(logs)
            print(f"Episode {episode} — Loss: {loss:.4f}, Reward: {reward:.2f}, Actions: {actions}")

        # Early stopping: if no improvement in best reward for 'patience' episodes
        if episode >= min_episodes and no_improve_count >= patience:
            print(f"Early stopping at episode {episode} (no improvement in {patience} episodes)")
            break

    save_logs(logs)
    return best, worst





if __name__ == "__main__":
    best, worst = create_study_guide(MARKDOWN_FILE)

    print("\nBest Study Guide:\n")
    print(f"Reward: {best["reward"]:.2f}")
    print(f"Actions: {best["actions"]}")
    print("\nStudy Guide:")
    print(best["response"])

    print("\nWorst Study Guide:\n")
    print(f"Reward: {worst["reward"]:.2f}")
    print(f"Actions: {worst["actions"]}")
    print("\nStudy Guide:")
    print(worst["response"])

