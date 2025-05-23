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

### Running the System
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

