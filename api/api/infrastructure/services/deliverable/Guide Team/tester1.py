"""
Tester file for the RL-based prompt engineering pipeline - MULTIPLE CONCEPTS
Fill in the parameters below and run this file to train the model.
"""

from RL_script import train_policy, bin_concepts, bin_learning_style

# ========== FILL IN YOUR PARAMETERS HERE ==========

# Concepts: List of dictionaries with "term" and "definition"
# These will be binned together as a group to find the best category
# Example: Linear Algebra concepts should all bin to "Linear Algebra"
concepts = [
    {
        "term": "derivative",
        "definition": "A derivative represents the rate of change of a function at any given point, measuring how the function's output changes as its input changes."
    },
    {
        "term": "limit",
        "definition": "A limit describes the value that a function approaches as the input approaches some value, forming the foundation for calculus concepts."
    },
    {
        "term": "chain rule",
        "definition": "The chain rule is a formula for computing the derivative of the composition of two or more functions."
    },
    {
        "term": "integration",
        "definition": "Integration is the reverse process of differentiation, used to find areas under curves and solve accumulation problems."
    }
]

# Learning Style: How does the student prefer to learn?
# Examples: "visual learner with diagrams", "step by step explanations", "real world examples", 
#           "socratic questioning", "hands-on practice", "analogies and metaphors"
learning_style = "visual learner with diagrams and step-by-step examples"

# Conversation: Previous conversation history as a list of strings
# Each string should represent one turn in the conversation (either student or AI response)
# Leave empty [] if this is the start of a new conversation
conversation = [
    "Student: I'm really struggling with calculus. There are so many related concepts and I can't see how they fit together.",
    "AI: I understand! Calculus can seem overwhelming because it builds on several interconnected ideas. Let's start by understanding how derivatives and limits relate to each other.",
    "Student: Yes, I know derivatives measure rate of change, but I don't get what limits have to do with it.",
    "AI: Great question! Limits are actually the foundation that makes derivatives possible. When we find a derivative, we're really finding the limit of a rate of change as the interval gets infinitely small.",
    "Student: That makes some sense, but then how does integration fit in? And what about all these rules like the chain rule?"
]

# ========== RUN THE TRAINING ==========

if __name__ == "__main__":
    print("="*60)
    print("RL-BASED PROMPT ENGINEERING TRAINING - MULTIPLE CONCEPTS")
    print("="*60)
    
    # Display the concepts
    print("Concepts to be trained on:")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept['term']}: {concept['definition'][:80]}...")
    
    print(f"\nLearning Style: {learning_style}")
    print(f"Conversation History: {len(conversation)} turns")
    print("="*60)
    
    # Show what bins these will map to
    print("\nBinning Results:")
    concept_bin = bin_concepts(concepts)
    learning_style_bin = bin_learning_style(learning_style)
    print(f"All concepts bin together to: {concept_bin}")
    print(f"Learning style bins to: {learning_style_bin}")
    print("="*60)
    
    # Train the policy with your parameters (using binned versions)
    policy, logs = train_policy(
        concepts=concepts,
        learning_style=learning_style_bin,
        conversation=conversation,
        filename="training_logs1.jsonl" 
    )
    
    print("\n" + "="*60)
    print("MULTIPLE CONCEPTS TRAINING COMPLETE")
    print("="*60)
    print("Files created:")
    print("- training_logs1.jsonl (real-time logs)")
    print(f"\nAll {len(concepts)} concepts were binned together as: {concept_bin}")