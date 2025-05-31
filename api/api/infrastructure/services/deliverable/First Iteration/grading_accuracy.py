import openai
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import asyncio
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=key)

# prompt
core_role = "College professor with a PhD and 20 years of teaching experience"
learning_style = "Examples and Analogies"
concepts = "Linear Algebra"
pedagogy_approach = "Inquiry-Based Learning"
question = "What is the rank of a matrix?"
prompt = (
   f"Your role is a {core_role} teaching a student about {concepts} who likes {learning_style}."
   f"Answer the specific question: {question} Then help the student learn the concept and other related concepts."
   f"Use {pedagogy_approach} to help the student learn."
   "Write your response in plain text with no emojis or symbols."
)

rubric = """1. clarity - is the explanation understandable & logically structured for the target learning style
score 1: contains multiple unclear sentences/poor structure
score 2: mostly understandable but includes 3+ confusing phrases/steps
score 3: understandable overall but has 1-2 unclear phrases or awkward transitions
score 4: clear except for 1 minor phrase/awkward transition
score 5: fully clear, logical, no confusing language

2. learning style fit - does the explanation align well with the requested learning style (ex. uses examples if example-based learner)
score 1: no visible attempt to tailor to the learning style
score 2: minimal tailoring, mentions style but does not change explanation
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
score 2: generic let me know style phrase without prompting further action
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

how to use the metrics: each metric is scored from 1 to 5 using the above criteria, multiply each score by its weight and sum for a final evaluation score"""

grading_prompt = f"Using the rubric, rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Please provide just the score as a number from 1-10. {prompt}"

# get response from openAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": core_role},
        {"role": "user", "content": prompt}
    ]
)

response = response.choices[0].message.content
print(response)

n_runs  = 100          
scores  = []

def create_histogram(vals):
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=range(1, 12), align="left", rwidth=0.8,
             color="steelblue", edgecolor="black")
    plt.title("Distribution of Grading Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xticks(range(1, 11))
    plt.grid(axis="y", alpha=0.7)
    plt.savefig("score_histogram.png")
    plt.close()
    print("Histogram saved as 'score_histogram.png'")

prompt = (
   f"Your role is a {core_role} teaching a student about {concepts} who likes {learning_style}."
   f"Answer the specific question: {question} Then help the student learn the concept and other related concepts."
   f"Use {pedagogy_approach} to help the student learn."
   "Write your response in plain text with no emojis or symbols."
)

grader = (
    "You are a grader who will use the following rubric to evaluate the response."
    "Grade the response on a scale of 1 to 10, where 1 is the worst and 10 is the best up to 2 decimal places."
    f"The prompt that was used to generate the response is: {prompt}."
)

async def main():
    for i in range(n_runs):
        try:
            grade = client.chat.completions.create(
                model="gpt-4o-mini",                
                messages=[
                    {"role": "system", "content": grader},
                    {"role": "user",   "content": grading_prompt}
                ],
                # temperature = 0.0
            )

            reply = grade.choices[0].message.content.strip()
            score = float(reply) 
            scores.append(score)
            print(f"{i+1}/{n_runs}: score={score}")

        except Exception as exc:
            print(f"Iteration {i+1} failed: {exc}")

    if scores:
        print(f"\nAverage score across {len(scores)} runs: "
              f"{sum(scores)/len(scores):.2f}")
        create_histogram(scores)

if __name__ == "__main__":
    asyncio.run(main())