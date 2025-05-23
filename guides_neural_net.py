import tensorflow as tf
import numpy as np
import json
from typing import List, Dict, Tuple, Any


# Bin 1: Concepts
concepts = [
    "Mathematics","Statistics", "Computer Science","Physics","Chemistry","Biology", "Environmental Science",
    "Psychology","Economics","Political Science", "Sociology",
    "History", "Literature", "Philosophy", "Art & Music",
    "Engineering","Medicine","Business","Law"
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
    "Self-taught expert and curriculum designer"
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

# Placeholder for rubric evaluation
def calculate_rubric_score(concept_idx: int, style_idx: int, role_idx: int, pedagogy_idx: int) -> float:
    """
    Simplified rubric scoring based on Sahil's metrics.
    Returns a weighted average score between 1 and 5.
    
    The metrics are:
    1. Understanding (30%) - How well the tutor understands and explains the concept
    2. Follows Learning Style (25%) - How well the tutor matches the specified learning style
    3. Asks Good Follow-Ups (15%) - Quality of follow-up questions
    4. Explains in a Helpful Way (15%) - Effectiveness of explanation
    5. Friendly and Encouraging (10%) - Tone and supportiveness
    6. Handles New Concepts Well (5%) - Ability to incorporate additional concepts
    """
 
    
    return weighted_score

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
    episodes = 200  # training episodes per permutation
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
                # forward pass inpiits 
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
                
                # log porb
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
            
            #updae weights
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
        
        results[(concept_name, style_name)] = {
            "role": core_roles[best_role],
            "pedagogy": pedagogy_approaches[best_ped],
            "score": calculate_rubric_score(
                c_idx, s_idx, best_role, best_ped
            )
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



