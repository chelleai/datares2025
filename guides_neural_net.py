import tensorflow as tf
import numpy as np
import json
from typing import List, Dict, Tuple, Any
import requests
import google.generativeai as genai
import time


# Bin 1: Concepts
concepts = [
    "Arithmetic & Number Theory",
    "Algebra"
    "Geometry",
    "Trigonometry",
    "Calculus",
    "Linear Algebra",
    "Discrete Mathematics",
    "Probability & Statistics",
    "Real & Complex Analysis",
    "Differential Equations",
    "Mathematical Logic & Set Theory",
    "Topology",
    "Mathematical Modeling",
    "Programming Fundamentals",
    "Data Structures & Algorithms",
    "Operating Systems",
    "Computer Architecture",
    "Databases",
    "Networking & Cybersecurity",
    "Software Engineering",
    "Web Development",
    "Artificial Intelligence",
    "Machine Learning",
    "Human-Computer Interaction",
    "Theory of Computation",
    "Computational Thinking",
    "Mechanical Engineering",
    "Electrical & Electronics Engineering",
    "Civil & Structural Engineering",
    "Computer Engineering",
    "Chemical Engineering",
    "Aerospace Engineering",
    "Environmental Engineering",
    "Materials Science",
    "Robotics & Automation",
    "Systems & Control Engineering",
    "Classical Mechanics",
    "Thermodynamics",
    "Electromagnetism",
    "Optics & Light",
    "Modern Physics (Relativity, Quantum Mechanics)",
    "Fluid Dynamics",
    "Astrophysics",
    "Nuclear Physics",
    "Geophysics",
    "Mathematical Physics",
    "Cell Biology",
    "Molecular Biology",
    "Genetics",
    "Evolution",
    "Physiology",
    "Ecology",
    "Microbiology",
    "Immunology",
    "Biotechnology",
    "Neuroscience",
    "Bioinformatics",
    "Inorganic Chemistry",
    "Organic Chemistry",
    "Physical Chemistry",
    "Analytical Chemistry",
    "Biochemistry",
    "Chemical Thermodynamics",
    "Reaction Kinetics",
    "Environmental Chemistry",
    "Developmental Psychology",
    "Cognitive Psychology",
    "Social Psychology",
    "Personality Theories",
    "Abnormal Psychology",
    "Educational Psychology",
    "Behavioral Neuroscience",
    "Classical Sociological Theory",
    "Contemporary Social Issues",
    "Social Stratification",
    "Urban Sociology",
    "Comparative Politics",
    "International Relations",
    "Public Policy & Administration",
    "Political Systems",
    "Political Economy",
    "Microeconomics",
    "Macroeconomics",
    "Econometrics",
    "Development Economics",
    "Game Theory",
    "Cultural Anthropology",
    "Archaeology",
    "Linguistic Anthropology",
    "Human Geography",
    "Cartography",
    "GIS (Geographic Information Systems)",
    "Logic & Reasoning",
    "Metaphysics",
    "Epistemology",
    "Ethics & Moral Philosophy",
    "Aesthetics",
    "Political Philosophy",
    "Philosophy of Mind",
    "Eastern Philosophy",
    "Existentialism",
    "World History",
    "Ancient Civilizations",
    "Medieval History",
    "Modern History",
    "History of Science",
    "Cultural & Intellectual History",
    "Comparative Religion",
    "Theology",
    "Mythology",
    "English Language & Composition",
    "Spanish Language",
    "French Language",
    "Mandarin Chinese",
    "German Language",
    "Arabic Language",
    "Latin & Classical Languages",
    "Linguistics",
    "Translation & Interpretation",
    "Poetry, Prose & Drama",
    "Literary Criticism & Theory",
    "Comparative Literature",
    "Creative Writing",
    "Drawing & Painting",
    "Sculpture",
    "Art History",
    "Design Principles",
    "Architecture",
    "Photography",
    "Film Studies",
    "Digital Media Art",
    "Theatre & Drama",
    "Acting Techniques",
    "Music Theory",
    "Music History",
    "Vocal & Instrumental Performance",
    "Dance & Choreography",
    "Education & Pedagogy",
    "Law & Legal Studies",
    "Journalism & Media Studies",
    "Communication Studies",
    "Public Health",
    "Urban Planning",
    "Gender Studies",
    "Cultural Studies",
    "Criminology",
    "Library & Information Science",
    "Business & Entrepreneurship",
    "Marketing & Advertising",
    "Finance & Accounting",
]

# Bin 2: Learning Styles
learning_styles = [
    "Verbal",
    "Visual",
    "Musical",
    "Physical",
    "Logical",
    "Social",
    "Solitary",
    "Combination"
]

core_roles = [
    "College professor with a PhD and 20 years of teaching experience",
    "Industry expert with practical experience",
    "Graduate student teaching assistant",
    "Experienced high school teacher",
    "Self-taught expert and curriculum designer",
    "Retired professor with a focus on innovative teaching methods",
    "Professional tutor specializing in personalized learning",
    "Educational consultant with a background in curriculum development",
    "Online course creator with a large following",
    "Research scientist with a passion for education",
    "Former principal with expertise in educational leadership",
    "Educational psychologist with a focus on student engagement",
    "STEM educator with experience in project-based learning",
    "Language instructor with a focus on immersive learning",
    "Special education teacher with expertise in inclusive teaching"
]

pedagogy_approaches = [
    "Inquiry-Based Learning",
    "Collaborative-Based Learning",
    "Inquiry-Based Learning",
    "Reflective Approach",
    "Problem-Based Learning"
]

# Mappings for model input/output
concept_to_idx: Dict[str, int] = {concept: i for i, concept in enumerate(concepts)}
idx_to_concept: Dict[int, str] = {i: concept for i, concept in enumerate(concepts)}

style_to_idx: Dict[str, int] = {style: i for i, style in enumerate(learning_styles)}
idx_to_style: Dict[int, str] = {i: style for i, style in enumerate(learning_styles)}


def calculate_rubric_score(concept_idx: int, style_idx: int, role_idx: int, pedagogy_idx: int) -> float:
    """
    Grades the prompt template using Gemini and the chelle ai rubric.
    Fills in the template with example values based on the provided indices, sends it to Gemini with the rubric, and returns the float score.
    Rate limits the API calls to avoid exceeding Gemini's quota.
    """
    genai.configure(api_key="AIzaSyCInmc6YCt6jhe4XC4eyzwcpQPJ9jzxRVY")
    model = genai.GenerativeModel('gemini-1.5-flash')

    concept = idx_to_concept.get(concept_idx, "[concept]")
    learning_style = idx_to_style.get(style_idx, "[learning_style]")
    core_role = core_roles[role_idx % len(core_roles)]
    pedagogy_approach = pedagogy_approaches[pedagogy_idx % len(pedagogy_approaches)]

    template = format_prompt(concept, learning_style, core_role, pedagogy_approach)

    rubric = """evaluation metrics for chelle ai

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
"""

    evaluation_prompt = f"""
You are an expert educational evaluator. Please use the following rubric to grade the provided guide template. For each metric, score from 1 to 5 as described. Multiply each score by its weight and sum for a final evaluation score. Return only the final score as a float (no explanation, no breakdown).

RUBRIC:
{rubric}

GUIDE TEMPLATE TO EVALUATE:
{template}

Return only the final score as a float.
"""

    max_retries = 5
    delay = 1  # seconds
    for attempt in range(max_retries):
        try:
            response = model.generate_content(evaluation_prompt)
            score_str = response.text.strip()
            time.sleep(delay)  # Rate limit: wait after each call
            return float(score_str)
        except Exception as e:
            # Check for rate limit error (customize this check as needed)
            if 'rate' in str(e).lower() or 'quota' in str(e).lower():
                print(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("Gemini API call failed:", e)
                return 0.0
    print("Gemini API call failed after retries due to rate limiting.")
    return 0.0

# Neural Network definition using TensorFlow/Keras
class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_concepts, num_styles, num_roles, num_pedagogies, 
                embedding_dim=32, hidden_dim=64):
        super(NeuralNetwork, self).__init__()
        
        # Embedding layers
        self.concept_embedding = tf.keras.layers.Embedding(
            num_concepts, embedding_dim, name="concept_embedding"
        )
        
        self.style_embedding = tf.keras.layers.Embedding(
            num_styles, embedding_dim, name="style_embedding"
        )
        
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(hidden_dim, activation='relu')
        
        # Output layers
        self.role_out = tf.keras.layers.Dense(num_roles)
        self.pedagogy_out = tf.keras.layers.Dense(num_pedagogies)

    def call(self, inputs):
        concept_idx, style_idx = inputs
        
        # Get embeddings and combine them
        concept_emb = self.concept_embedding(concept_idx)
        style_emb = self.style_embedding(style_idx)
        combined = tf.concat([concept_emb, style_emb], axis=-1)
        
        # Forward pass
        hidden = self.hidden(combined)
        role_logits = self.role_out(hidden)
        pedagogy_logits = self.pedagogy_out(hidden)
        
        return role_logits, pedagogy_logits

# Function to select unique (concept, style) permutations
def get_target_permutations(num_permutations: int = 20) -> List[Tuple[str, str]]:
    all_pairs: List[Tuple[str, str]] = []
    for concept_name in concepts:
        for style_name in learning_styles:
            all_pairs.append((concept_name, style_name))
    
    # Shuffle for variety if we have more than num_permutations, then pick
    np.random.shuffle(all_pairs)
    return all_pairs[:num_permutations]

# Function to format prompt using selected parameters
def format_prompt(concept, learning_style, core_role, pedagogy_approach, 
                 question="What is this concept and how is it used?"):
    
    prompt = f"""Your role is a {core_role} teaching a student about {concept} who likes {learning_style}. 
Answer the specific question: {question} 
Then help the student learn the concept and other related concepts. 
Use {pedagogy_approach} to help the student learn. 

1. Show clear understanding of the concept with accurate explanations.
2. Present information using the student's preferred {learning_style} learning style throughout your response.
3. Include follow-up questions to check understanding and guide further learning.
4. Explain concepts in a helpful, structured way that builds from simple to complex.
5. Be warm, friendly, and encouraging with a supportive tone.
6. Connect this concept to at least one related concept, explaining the relationship between them.

Write your response in plain text with no emojis or symbols."""
    return prompt


def train_rl_agent():
    """Train the RL agent to find optimal prompt parameters."""
    # Hyperparameters
    episodes = 10  # training episodes per permutation
    lr = 0.001
    emb_dim = 16  
    h_dim = 32
    n_permutations = 20
    
    # permutations to train on
    perms = get_target_permutations(n_permutations)
    
    print("Found " + str(len(perms)) + " permutations for training.")
    for i, (concept, style) in enumerate(perms):
        print("Perm " + str(i+1) + "/" + str(len(perms)) + ": " + concept + ", " + style)

    # Track results
    results = {}
    training_results = []

    # Train on each permutation
    for idx, (concept_name, style_name) in enumerate(perms):
        print("Training permutation " + str(idx + 1) + ": " + concept_name + ", " + style_name + " ---")

        # Get indices
        c_idx = concept_to_idx[concept_name]
        s_idx = style_to_idx[style_name]

        # Create network and optimizer
        net = NeuralNetwork(
            len(concepts),
            len(learning_styles),
            len(core_roles),
            len(pedagogy_approaches),
            emb_dim,
            h_dim
        )
        
        opt = tf.keras.optimizers.Adam(lr)

        # Training loop
        for ep in range(episodes):
            with tf.GradientTape() as tape:
                # forward pass inputs 
                c_tensor = tf.convert_to_tensor([c_idx], dtype=tf.int32)
                s_tensor = tf.convert_to_tensor([s_idx], dtype=tf.int32)
                role_logits, ped_logits = net([c_tensor, s_tensor])
                
                # get action probabilities and sample
                role_probs = tf.nn.softmax(role_logits, axis=-1)
                ped_probs = tf.nn.softmax(ped_logits, axis=-1)
                
                role_dist = tf.random.categorical(tf.math.log(role_probs), 1)
                ped_dist = tf.random.categorical(tf.math.log(ped_probs), 1)
                
                # get chosen actions
                role_idx = role_dist[0, 0].numpy()
                ped_idx = ped_dist[0, 0].numpy()
                
                # calc log probs
                log_prob_role = tf.math.log(role_probs[0, role_idx])
                log_prob_ped = tf.math.log(ped_probs[0, ped_idx])
                
                # log prob
                total_log_prob = log_prob_role + log_prob_ped
                
                # reward is the rubric score
                reward = calculate_rubric_score(
                    c_idx, s_idx, int(role_idx), int(ped_idx)
                )
                
                # we REINFORCE loss function 
                loss = -total_log_prob * reward

                # Log results for this episode
                episode_result = {
                    "episode": ep + 1,
                    "reward": float(reward),
                    "actions": {
                        "core_role": core_roles[int(role_idx)],
                        "pedagogical_strategy": pedagogy_approaches[int(ped_idx)],
                        "answer_format": "structured",
                        "memory_mode": "active",
                        "tone_style": "friendly"
                    },
                    "prompt": format_prompt(
                        concept_name,
                        style_name,
                        core_roles[int(role_idx)],
                        pedagogy_approaches[int(ped_idx)]
                    ),
                    "response": "Training in progress..."
                }
                training_results.append(episode_result)
            
            # update weights
            grads = tape.gradient(loss, net.trainable_variables)
            opt.apply_gradients(zip(grads, net.trainable_variables))
            
            if ep == episodes - 1:
                print("  Training complete. Final reward: " + str(round(reward, 2)) + ", Loss: " + str(round(loss.numpy(), 2)))
        
        # best parameters after training
        c_tensor = tf.convert_to_tensor([c_idx], dtype=tf.int32)
        s_tensor = tf.convert_to_tensor([s_idx], dtype=tf.int32)
        
        role_logits, ped_logits = net([c_tensor, s_tensor])
        
        best_role = int(tf.argmax(role_logits, axis=-1)[0])
        best_ped = int(tf.argmax(ped_logits, axis=-1)[0])
        
        # Call Gemini once per permutation to score the best parameters
        final_score = calculate_rubric_score(
            c_idx, s_idx, best_role, best_ped
        )
        
        results[(concept_name, style_name)] = {
            "role": core_roles[best_role],
            "pedagogy": pedagogy_approaches[best_ped],
            "score": final_score
        }
        
        sample = format_prompt(
            concept_name, 
            style_name, 
            core_roles[best_role], 
            pedagogy_approaches[best_ped]
        )

        print("  Best params for " + concept_name + " & " + style_name + ":")
        print("    Role: " + core_roles[best_role])
        print("    Pedagogy: " + pedagogy_approaches[best_ped])
        print("    (Score: " + str(round(results[(concept_name, style_name)]['score'], 2)) + ")")
        print("  Sample prompt:")
        print("    " + sample)

        # Save results to JSON file after each permutation
        with open('results.json', 'w') as f:
            json.dump(training_results, f, indent=2)

    print("Best Parameters")
    for (c, s), params in results.items():
        print(c + ", " + s)
        print("  Role: " + params['role'])
        print("  Pedagogy: " + params['pedagogy'])
        print("  (Score: " + str(round(params['score'], 2)) + ")")
        print("-" * 20)
    
    # Final save of results
    with open('results.json', 'w') as f:
        json.dump(training_results, f, indent=2)


if __name__ == "__main__":
    train_rl_agent()



