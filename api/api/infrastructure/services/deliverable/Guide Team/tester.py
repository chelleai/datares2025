"""
Tester file for the RL-based prompt engineering pipeline.
Fill in the parameters below and run this file to train the model.
"""

from RL_script import train_policy, bin_concept, bin_learning_style

# ========== FILL IN YOUR PARAMETERS HERE ==========

# Concept: What topic/subject are you teaching?
# Examples: "linear algebra", "photosynthesis", "World War 2", "machine learning", "calculus derivatives"
concept = "calculus derivatives"

# Learning Style: How does the student prefer to learn?
# Examples: "visual learner with diagrams", "step by step explanations", "real world examples", 
#           "socratic questioning", "hands-on practice", "analogies and metaphors"
learning_style = "visual learner with diagrams and step-by-step examples"

# Conversation: Previous conversation history as a list of strings
# Each string should represent one turn in the conversation (either student or AI response)
# Leave empty [] if this is the start of a new conversation
conversation = [
    "Student: I'm really struggling with derivatives. Can you help me understand what they mean?",
    "AI: Of course! A derivative represents the rate of change of a function at any given point. Think of it like the speedometer in your car - it tells you how fast you're going at that exact moment.",
    "Student: That makes some sense, but I'm still confused about how to actually calculate them.",
    "AI: Let's start with the basic definition and work through some simple examples step by step.",
    "Student: What is the derivative of x^2?"
]

# ========== RUN THE TRAINING ==========

if __name__ == "__main__":
    print("="*50)
    print("RL-BASED PROMPT ENGINEERING TRAINING")
    print("="*50)
    print(f"Concept: {concept}")
    print(f"Learning Style: {learning_style}")
    print(f"Conversation History: {len(conversation)} turns")
    print("="*50)
    
    # If you want to bin the concept and learning style, uncomment the lines below
    concept = bin_concept(concept)
    learning_style = bin_learning_style(learning_style)

    # Train the policy with your parameters
    policy, logs = train_policy(
        concept=concept,
        learning_style=learning_style,
        conversation=conversation
    )
    
    print("\n" + "="*50)
    print("Training Complete")
    print("="*50)
    print("Check the 'checkpoints' folder for saved models")
    print("Check 'training_logs.jsonl' for detailed training logs")