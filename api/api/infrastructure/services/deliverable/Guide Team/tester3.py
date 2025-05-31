"""
Tester file for the RL-based prompt engineering pipeline - WORLD HISTORY/SOCIAL SCIENCES
Multiple concepts that should bin together to History category.
"""

from RL_script import train_policy, bin_concepts, bin_learning_style

# ========== WORLD HISTORY/SOCIAL SCIENCES TEST - MULTIPLE CONCEPTS ==========

# Concepts: List of WWI-related dictionaries with "term" and "definition"
# These should all bin together to "World History" or similar Social Sciences category
concepts = [
    {
        "term": "imperialism",
        "definition": "The policy of extending a country's power and influence through diplomacy or military force, often involving the acquisition of colonies and territories."
    },
    {
        "term": "alliance system",
        "definition": "A network of mutual defense agreements between European powers in the early 20th century that divided Europe into opposing camps."
    },
    {
        "term": "nationalism",
        "definition": "A political ideology emphasizing loyalty and devotion to one's nation, often involving the desire for national independence or unification."
    },
    {
        "term": "trench warfare",
        "definition": "A type of combat in which opposing troops fight from trenches facing each other, characterized by little territorial gain despite heavy casualties."
    },
    {
        "term": "total war",
        "definition": "A conflict in which countries devote all their resources to the war effort, blurring the distinction between civilian and military targets."
    },
    {
        "term": "armistice",
        "definition": "An agreement to stop fighting temporarily or permanently, typically while a peace treaty is negotiated between warring parties."
    }
]

# Learning Style: Creative storytelling approach - completely different from previous styles
learning_style = "creative storytelling with narratives and character perspectives"

# Conversation: History-focused conversation with narrative elements
conversation = [
    "Student: World War I seems so complicated. There were so many countries involved, I can't keep track of why it even started.",
    "AI: I understand the confusion! Let me tell you the story like a dramatic play. Picture Europe in 1914 as a powder keg - tensions had been building for years through a web of alliances, like a complicated friendship drama where everyone had picked sides.",
    "Student: But what was the actual spark? What made it explode into war?",
    "AI: The spark was the assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo. Imagine a domino effect - Austria-Hungary blamed Serbia, Russia backed Serbia, Germany backed Austria-Hungary, France was allied with Russia, and Britain was drawn in to protect Belgium.",
    "Student: That's a lot of dominoes! But why did this one assassination cause such a massive war? Surely people get assassinated without starting world wars.",
    "AI: You're absolutely right! The assassination was just the trigger, not the root cause. Think of it like the final straw that broke the camel's back. The real causes were deeper - imperialism, militarism, and nationalism had been building tension for decades.",
    "Student: I see, so it was like all these underlying tensions were waiting for an excuse to explode. But how did it get so devastating? Why couldn't they just have a small war and be done with it? And what made it end?"
]

# ========== RUN THE TRAINING ==========

if __name__ == "__main__":
    print("="*60)
    print("RL-BASED PROMPT ENGINEERING TRAINING - WORLD HISTORY CONCEPTS")
    print("="*60)

    # Display the concepts
    print("World War I concepts to be trained on:")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept['term']}: {concept['definition'][:65]}...")
    
    print(f"\nLearning Style: {learning_style}")
    print(f"Conversation History: {len(conversation)} turns")
    print("="*60)
    
    # Show what bins these will map to
    print("\nBinning Results:")
    concept_bin = bin_concepts(concepts)
    learning_style_bin = bin_learning_style(learning_style)
    print(f"All history concepts bin together to: {concept_bin}")
    print(f"Learning style bins to: {learning_style_bin}")
    print("="*60)

    # Train the policy with your parameters (using binned versions)
    policy, logs = train_policy(
        concepts=concepts,
        learning_style=learning_style_bin,
        conversation=conversation,
        filename="training_logs3.jsonl",
    )

    print("\n" + "="*60)
    print("WORLD HISTORY TRAINING COMPLETE")
    print("="*60)
    print("Files created:")
    print("- training_logs3.jsonl (real-time logs)")
    print(f"\nAll {len(concepts)} WWI concepts were binned together as: {concept_bin}")