# Guides Team Reinforcement Learning Agent

This project implements a reinforcement learning (RL) system that optimizes educational prompts based on subject matter and learning styles.

## Overview

The system uses a neural network to learn optimal combinations of educational parameters:
- Core instructor roles
- Pedagogy approaches
- Response formats

These parameters are optimized for different combinations of:
- Subject matter (e.g., Linear Algebra, Computer Science)
- Learning styles (e.g., Visual, Examples and Analogies)

## Setup

### Requirements

To run this code, you'll need:
- Python 3.7+
- PyTorch or TensorFlow (two implementations available)
- NumPy

Install dependencies:

```bash
# For PyTorch implementation
pip install torch numpy

# For TensorFlow implementation
pip install tensorflow numpy
```

### Running the System

To train the RL agent:

```bash
# For PyTorch implementation
python guides_rl_agent.py

# For TensorFlow implementation
python guides_rl_agent_tf.py
```

## How It Works

1. The system defines two "bins" of input parameters:
   - Concepts/Subjects (what is being taught)
   - Learning Styles (how the student prefers to learn)

2. For each combination of concept and learning style, the RL agent learns to select:
   - The best teacher/instructor role
   - The optimal pedagogy approach
   - The most effective response format

3. The training process uses a simulated scoring function that will eventually be replaced with actual evaluations using a detailed rubric.

4. After training, the system outputs optimal prompt parameters for each concept-style combination.

## Implementations

Two implementations are provided:
- `guides_rl_agent.py` - PyTorch implementation
- `guides_rl_agent_tf.py` - TensorFlow implementation

Both implementations provide identical functionality but use different deep learning frameworks.

## Prompt Template

The system uses a prompt template of the form:

```
Your role is a [CORE_ROLE] teaching a student about [CONCEPT] who likes [LEARNING_STYLE].
Answer the specific question: [QUESTION] Then help the student learn the concept and other related concepts.
Use [PEDAGOGY_APPROACH] to help the student learn.
Write your response in plain text with no emojis or symbols.
```

## Extending the System

To expand this system:
1. Add more concepts/subjects to the `CONCEPTS` list
2. Add more learning styles to the `LEARNING_STYLES` list
3. Refine the action space (`CORE_ROLES`, `PEDAGOGY_APPROACHES`, `RESPONSE_FORMATS`) 
4. Implement a more accurate scoring function by replacing the placeholder with actual LLM generation and rubric evaluation