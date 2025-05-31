"""
Tester file for the RL-based prompt engineering pipeline - CHEMISTRY/PHYSICAL SCIENCES
Multiple concepts that should bin together to Chemistry category.
"""

from RL_script import train_policy, bin_concepts, bin_learning_style

# ========== CHEMISTRY/PHYSICAL SCIENCES TEST - MULTIPLE CONCEPTS ==========

# Concepts: List of chemistry-related dictionaries with "term" and "definition"
# These should all bin together to "General Chemistry" or similar Physical Sciences category
concepts = [
    {
        "term": "ionic bond",
        "definition": "A chemical bond formed through the electrostatic attraction between oppositely charged ions, typically between a metal and a non-metal."
    },
    {
        "term": "covalent bond", 
        "definition": "A chemical bond formed when two atoms share one or more pairs of electrons, typically between non-metal atoms."
    },
    {
        "term": "electronegativity",
        "definition": "A measure of an atom's ability to attract and hold electrons when forming chemical bonds."
    },
    {
        "term": "molecular geometry",
        "definition": "The three-dimensional arrangement of atoms in a molecule, determined by electron pair repulsion and bonding patterns."
    },
    {
        "term": "polar molecule",
        "definition": "A molecule with an uneven distribution of electron density, resulting in partial positive and negative charges on different parts of the molecule."
    }
]

# Learning Style: Real-world examples and analogies - different from calculus test
learning_style = "learn through real-world analogies and concrete examples"

# Conversation: Chemistry-focused conversation with different complexity
conversation = [
    "Student: I don't understand why atoms bond together. What's the point?",
    "AI: Great question! Atoms bond because they want to achieve a stable electron configuration, like having a full outer shell of electrons. It's like people forming friendships - they do it because it makes them more stable and happy.",
    "Student: But how do they actually stick together? Is it like glue?",
    "AI: Not quite like glue! There are different types of bonds. Ionic bonds form when one atom gives electrons to another, like a generous friend sharing. Covalent bonds form when atoms share electrons equally, like roommates sharing groceries.",
    "Student: I'm still confused about ionic vs covalent bonds. Can you give me a real example?",
    "AI: Absolutely! Salt (NaCl) is ionic - sodium gives an electron to chlorine. It's like sodium saying 'here, take this electron, you need it more than I do.' Water (H2O) is covalent - hydrogen and oxygen share electrons equally.",
    "Student: Why does this sharing vs giving matter? What's the difference in how they behave? And what about these molecular shapes I keep hearing about?"
]

# ========== RUN THE TRAINING ==========

if __name__ == "__main__":
    print("="*60)
    print("RL-BASED PROMPT ENGINEERING TRAINING - CHEMISTRY CONCEPTS")
    print("="*60)
    
    # Display the concepts
    print("Chemistry concepts to be trained on:")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept['term']}: {concept['definition'][:70]}...")
    
    print(f"\nLearning Style: {learning_style}")
    print(f"Conversation History: {len(conversation)} turns")
    print("="*60)

    # Show what bins these will map to
    print("\nBinning Results:")
    concept_bin = bin_concepts(concepts)
    learning_style_bin = bin_learning_style(learning_style)
    print(f"All chemistry concepts bin together to: {concept_bin}")
    print(f"Learning style bins to: {learning_style_bin}")
    print("="*60)

    # Train the policy with your parameters (using binned versions)
    policy, logs = train_policy(
        concepts=concepts,
        learning_style=learning_style_bin,
        conversation=conversation,
        filename="training_logs2.jsonl",
    )

    print("\n" + "="*60)
    print("CHEMISTRY TRAINING COMPLETE")
    print("="*60)
    print("Files created:")
    print("- training_logs2.jsonl (real-time logs)")
    print(f"\nAll {len(concepts)} chemistry concepts were binned together as: {concept_bin}")