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
CONTEXT_PROMPT = "Please give me the context of the following text. You should provide the topic of the text and the type of source (e.g. textbook, interview, news article, etc.). The text is:\n{text}"
TERM_CITATION_PROMPT = "Given this markdown file, extract the key terms needed to understand the topic of the markdown file. For each term, get a list of quoted citations from the markdown file that relate to the term. Your return output should make clear which citations belong to each term. Only include the terms and citations, and no outside text. The text is:\n{text}"

# prompts to classify the terms+citations into bins
CONTEXT_BIN_PROMPT = "You are given a list of context bins that specify the source of a file. Given a file containing the context give the most logical bin that fits the text provided. Only return the bin name, with no extra whitespace and new lines.\nHere is the list of context bins:{context_bins}\nHere is the list of terms and citations:{context}"
SUBJECT_BIN_PROMPT = "You are given a list of subject bins that specify the topic of a file. Given a list of terms and citations, give the most logical subject it falls under, choosing a more specific subject if available, and only if all concepts contained in the text fit under that subject. Only return the bin name, with no extra whitespace and new lines. Here is the list of subject bins:{subject_bins}\nHere is the text containing the terms and citations:{terms_citations}"

# prompt for getting LLM to evaluate the output
EVALUATION_PROMPT = "{eval_metric}\n\nContext:{context}\nSubject:{subject}\nStudy Guide:\n{response}\n\nBe as harsh and nitpicky as you can when grading each individual score. You should not give out 5s unless the response is completely perfect. Please provide a final score between 0 and 100 as a single number. Calculate the final evaluation score at the end of your response. The final score should be the last number you write."



# bins
CONTEXT_BINS = ["Textbook", "Interview", "Song Lyrics", "News Article", "Blog Article", "Academic Paper", "Research Paper", "Dictionary", "Encyclopedia", "Documentary", "Fiction Book", "Nonfiction Book", "Conversation"]
SUBJECT_BINS = [
    "Math", "Statistics", "Calculus", "Linear Algebra", "Geometry", "Algebra", "Arithmetic", "Discrete Mathematics",
    "English", "Literature", "English Literature", "Grammar", "Poetry", "Journalism", "Shakespeare",
    "Science", "Biology", "Chemistry", "Astronomy", "Earth Science", "Physics", "Biochemistry", "Environmental Science", "Psychology", "Geography", "Geology", "Anatomy", "Sociology",
    "Economics", "Business", "Communications", "Political Science", "Government", "Management", "Accounting", "Finance", "Public Affairs",
    "Social Studies", "History", "American History", "European History", "World History", "Asian History", "African History",
    "Foreign Language", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Hindi", "Portuguese", "Italian", "Latin",
    "Technology", "Computer Science", "Data Science", "Architecture",
    "Engineering", "Software Engineering", "Chemical Engineering", "Bioengineering", "Electrical Engineering", "Industrial Engineering", "Mechanical Engineering", "Civil Engineering", "Aerospace Engineering",
    "Medicine", "Law", "Education", "Life Science", "Physical Science", "Social Science", "Humanities",
    "Art", "Design", "Marketing", "Performing Arts", "Philosophy", "Religion", "Linguistics", "Ethics",
    "Music", "Physical Education", "Health", "Cooking", "Fashion",
    "Pop Culture", "TV Shows", "Movies", "Celebrities", "Film", "Lifestyle",
    "Transportation", "Social Media", "Memes", "Video Games", "Games", "Sports", "Animals", "Food", "Holidays", "Gardening", "Hospitality", "Everyday Life", "Jobs"
]


# test markdown file
MARKDOWN_FILE = """
Chapter I - The Old Sea-dog at the 'Admiral Benbow'
Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having asked me to
write down the whole particulars about Treasure Island, from the beginning to the
end, keeping nothing back but the bearings of the island, and that only because
there is still treasure not yet lifted, I take up my pen in the year of grace 17_ and go
Treasure Island Robert Louis Stevenson Page 2/142
back to the time when my father kept the Admiral Benbow inn and the brown old
seaman with the sabre cut first took up his lodging under our roof.
I remember him as if it were yesterday, as he came plodding to the inn door, his
sea- chest following behind him in a hand-barrow — a tall, strong, heavy, nut-brown
man, his tarry pigtail falling over the shoulder of his soiled blue coat, his hands
ragged and scarred, with black, broken nails, and the sabre cut across one cheek, a
dirty, livid white. I remember him looking round the cover and whistling to himself
as he did so, and then breaking out in that old sea-song that he sang so often
afterwards:
“Fifteen men on the dead man’s chest — Yo-ho-ho, and a bottle of rum!”
in the high, old tottering voice that seemed to have been tuned and broken at the
capstan bars. Then he rapped on the door with a bit of stick like a handspike that
he carried, and when my father appeared, called roughly for a glass of rum. This,
when it was brought to him, he drank slowly, like a connoisseur, lingering on the
taste and still looking about him at the cliffs and up at our signboard.
“This is a handy cove,” says he at length; “and a pleasant sittyated grog-shop. Much
company, mate?”
My father told him no, very little company, the more was the pity.
“Well, then,” said he, “this is the berth for me. Here you, matey,” he cried to the
man who trundled the barrow; “bring up alongside and help up my chest. I’ll stay
here a bit,” he continued. “I’m a plain man; rum and bacon and eggs is what I want,
and that head up there for to watch ships off. What you mought call me? You
mought call me captain. Oh, I see what you’re at — there”; and he threw down three
or four gold pieces on the threshold. “You can tell me when I’ve worked through
that,” says he, looking as fierce as a commander.
And indeed bad as his clothes were and coarsely as he spoke, he had none of the
German
appearance of a man who sailed before the mast, but seemed like a mate or skipper
accustomed to be obeyed or to strike. The man who came with the barrow told us
the mail had set him down the morning before at the Royal George, that he had
inquired what inns there were along the coast, and hearing ours well spoken of, I
suppose, and described as lonely, had chosen it from the others for his place of
residence. And that was all we could learn of our guest.
He was a very silent man by custom. All day he hung round the cove or upon the
cliffs with a brass telescope; all evening he sat in a corner of the parlour next the
fire and drank rum and water very strong. Mostly he would not speak when spoken
Treasure Island Robert Louis Stevenson Page 3/142
to, only look up sudden and fierce and blow through his nose like a fog-horn; and we
and the people who came about our house soon learned to let him be. Every day
when he came back from his stroll he would ask if any seafaring men had gone by
along the road. At first we thought it was the want of company of his own kind that
made him ask this question, but at last we began to see he was desirous to avoid
them. When a seaman did put up at the Admiral Benbow (as now and then some
did, making by the coast road for Bristol) he would look in at him through the
curtained door before he entered the parlour; and he was always sure to be as silent
as a mouse when any such was present. For me, at least, there was no secret about
the matter, for I was, in a way, a sharer in his alarms. He had taken me aside one
day and promised me a silver fourpenny on the first of every month if I would only
keep my “weather-eye open for a seafaring man with one leg” and let him know the
moment he appeared. Often enough when the first of the month came round and I
applied to him for my wage, he would only blow through his nose at me and stare
me down, but before the week was out he was sure to think better of it, bring me my
four-penny piece, and repeat his orders to look out for “the seafaring man with one
leg.”
How that personage haunted my dreams, I need scarcely tell you. On stormy nights,
when the wind shook the four corners of the house and the surf roared along the
cove and up the cliffs, I would see him in a thousand forms, and with a thousand
diabolical expressions. Now the leg would be cut off at the knee, now at the hip; now
he was a monstrous kind of a creature who had never had but the one leg, and that
in the middle of his body. To see him leap and run and pursue me over hedge and
ditch was the worst of nightmares. And altogether I paid pretty dear for my
monthly fourpenny piece, in the shape of these abominable fancies.
But though I was so terrified by the idea of the seafaring man with one leg, I was
far less afraid of the captain himself than anybody else who knew him. There were
nights when he took a deal more rum and water than his head would carry; and
then he would sometimes sit and sing his wicked, old, wild sea-songs, minding
nobody; but sometimes he would call for glasses round and force all the trembling
company to listen to his stories or bear a chorus to his singing. Often I have heard
the house shaking with “Yo-ho-ho, and a bottle of rum,” all the neighbours joining in
for dear life, with the fear of death upon them, and each singing louder than the
other to avoid remark. For in these fits he was the most overriding companion ever
known; he would slap his hand on the table for silence all round; he would fly up in
a passion of anger at a question, or sometimes because none was
put, and so he judged the company was not following his story. Nor would he allow
anyone to leave the inn till he had drunk himself sleepy and reeled off to bed.
Treasure Island Robert Louis Stevenson Page 4/142
His stories were what frightened people worst of all. Dreadful stories they were —
about hanging, and walking the plank, and storms at sea, and the Dry Tortugas,
and wild deeds and places on the Spanish Main. By his own account he must have
lived his life among some of the wickedest men that God ever allowed upon the sea,
and the language in which he told these stories shocked our plain country people
almost as much as the crimes that he described. My father was always saying the
inn would be ruined, for people would soon cease coming there to be tyrannized over
and put down, and sent shivering to their beds; but I really believe his presence did
us good. People were frightened at the time, but on looking back they rather liked it;
it was a fine excitement in a quiet country life, and there was even a party of the
younger men who pretended to admire him, calling him a “true sea-dog” and a “real
old salt” and such like names, and saying there was the sort of man that made
England terrible at sea. In one way, indeed, he bade fair to ruin us, for he kept on
staying week after week, and at last month after month, so that all the money had
been long exhausted, and still my father never plucked up the heart to insist on
having more. If ever he mentioned it, the captain blew through his nose so loudly
that you might say he roared, and stared my poor father out of the room. I have
seen him wringing his hands after such a rebuff, and I am sure the annoyance and
the terror he lived in must have greatly hastened his early and unhappy death.
All the time he lived with us the captain made no change whatever in his dress but
to buy some stockings from a hawker. One of the cocks of his hat having fallen
down, he let it hang from that day forth, though it was a great annoyance when it
blew. I remember the appearance of his coat, which he patched himself upstairs in
his room, and which, before the end, was nothing but patches. He never wrote or
received a letter, and he never spoke with any but the neighbours, and with these,
for the most part, only when drunk on rum. The great sea-chest none of us had ever
seen open.
He was only once crossed, and that was towards the end, when my poor father was
far gone in a decline that took him off. Dr. Livesey came late one afternoon to see
the patient, took a bit of dinner from my mother, and went into the parlour to
smoke a pipe until his horse should come down from the hamlet, for we had no
stabling at the old Benbow. I followed him in, and I remember observing the
contrast the neat, bright doctor, with his powder as white as snow and his bright,
black eyes and pleasant manners, made with the coltish country folk, and above all,
with that filthy, heavy, bleared scarecrow of a pirate of ours, sitting, far gone in
rum, with his arms on the table. Suddenly he — the captain, that is — began to
pipe up his eternal song:
“Fifteen men on the dead man’s chest — Yo-ho-ho, and a bottle of rum! Drink and
the devil had done for the rest — Yo-ho-ho, and a bottle of rum!”
Treasure Island Robert Louis Stevenson Page 5/142
At first I had supposed “the dead man’s chest” to be that identical big box of his
upstairs in the front room, and the thought had been mingled in my nightmares
with that of the one-legged seafaring man. But by this time we had all long ceased
to pay any particular notice to the song; it was new, that night, to nobody but Dr.
Livesey, and on him I observed it did not produce an agreeable effect, for he looked
up for a moment quite angrily before he went on with his talk to old Taylor, the
gardener, on a new cure for the rheumatics. In the meantime, the captain gradually
brightened up at his own music, and at last flapped his hand upon the table before
him in a way we all knew to mean silence. The voices stopped at once, all but Dr.
Livesey’s; he went on as before speaking clear and kind and drawing briskly at his
pipe between every word or two. The captain glared at him for a while, flapped his
hand again, glared still harder, and at last broke out with a villainous, low oath,
“Silence, there, between decks!”
“Were you addressing me, sir?” says the doctor; and when the ruffian had told him,
with another oath, that this was so, “I have only one thing to say to you, sir,” replies
the doctor, “that if you keep on drinking rum, the world will soon be quit of a very
dirty scoundrel!”
The old fellow’s fury was awful. He sprang to his feet, drew and opened a sailor’s
clasp-knife, and balancing it open on the palm of his hand, threatened to pin the
doctor to the wall.
The doctor never so much as moved. He spoke to him as before, over his shoulder
and in the same tone of voice, rather high, so that all the room might hear, but
perfectly calm and steady: “If you do not put that knife this instant in your pocket, I
promise, upon my honour, you shall hang at the next assizes.”
Then followed a battle of looks between them, but the captain soon knuckled under,
put up his weapon, and resumed his seat, grumbling like a beaten dog.
“And now, sir,” continued the doctor, “since I now know there’s such a fellow in my
district, you may count I’ll have an eye upon you day and night. I’m not a doctor
only; I’m a magistrate; and if I catch a breath of complaint against you, if it’s only
for a piece of incivility like tonight’s, I’ll take effectual means to have you hunted
down and routed out of this. Let that suffice.”
Soon after, Dr. Livesey’s horse came to the door and he rode away, but the captain
held his peace that evening, and for many evenings to come.
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







def encode_bins(context_bin: str, subject_bin: str) -> torch.Tensor:
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
    
    encoding = encode_bins(context_bin, subject_bin).to(DEVICE)

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
    reward = extract_score(evaluation)

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
    
    # track best and worst outputs
    logs = []
    best: Dict[str, object] = {
        "reward": -1.0,
        "actions": None,
        "prompt": None,
        "response": None,
        "evaluation": None
    }
    worst: Dict[str, object] = {
        "reward": 101.0,
        "actions": None,
        "prompt": None,
        "response": None,
        "evaluation": None
    }

    reward_history = []
    min_episodes = 10
    std_threshold = 2 # triggers early stopping
    max_episodes = 50

    # training loop
    for episode in range(1, max_episodes+1):
        loss, reward, context, subject, actions, prompt, response, evaluation = train_step(policy, optimizer, markdown)
        print(f"Episode {episode} - Loss: {loss:.4f}, Reward: {reward:.2f}")

        # update best results
        if reward > best["reward"]:
            best["reward"] = reward
            best["actions"] = actions
            best["prompt"] = prompt
            best["response"] = response
            best["evaluation"] = evaluation
            print(f"New best reward: {best["reward"]:.2f}")
        
        # update worst result
        if reward < worst["reward"]:
            worst["reward"] = reward
            worst["actions"] = actions
            worst["prompt"] = prompt
            worst["response"] = response
            worst["evaluation"] = evaluation

        reward_history.append(reward)

        # log each episode
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

        # save logs periodically
        if episode % 5 == 0:
            save_logs(logs)
            print(f"Episode {episode} — Loss: {loss:.4f}, Reward: {reward:.2f}, Actions: {actions}")

        # stop early if reward > 2 std above mean
        if episode >= min_episodes:
            mean_reward = np.mean(reward_history)
            std = np.std(reward_history)
            if reward > mean_reward + (std_threshold * std):
                print(f"Stopped early at episode {episode}")
                print(f"Found exceptional result: {reward:.2f} (mean: {mean_reward:.2f}, std: {std:.2f})")
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


