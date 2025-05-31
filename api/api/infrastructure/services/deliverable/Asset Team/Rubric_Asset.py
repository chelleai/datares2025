EVALUATION_METRIC = """
1. Accuracy (0.20)
Are the extracted terms and the content that was generated, accurate, and relevant?
- 0 pt: Contains any factual or major errors
- 2 pts: No factual errors but 1-2 important details missing or imprecise
- 5 pts: Fully correct, complete, and precise

2. Completeness (0.20)
Does the output capture all relevant terms from the content, with no significant omissions?
- 0 pt: Less than half (<50%) of relevant terms are extracted
- 2 pts: Less than all relevant terms are extracted
- 5 pts: All relevant terms are extracted; no omissions

3. Citation Quality (0.15)
Are citations for each term accurate and appropriately matched to the content?
- 0 pt: Most citations are missing, irrelevant, or incorrect
- 2 pts: Some citations are accurate, but major mismatches or missing references remain
- 5 pts: All citations are accurate and included in the response

4. Relevance (0.10)
Are only meaningful and contextually appropriate terms extracted?
- 0 pt: Many extracted terms are generic, irrelevant, or off-topic
- 2 pts: Several extracted terms lack relevance to the context
- 5 pts: All extracted terms are contextually relevant and meaningful

5. Clarity (0.10)
Is the output formatted clearly and presented in an understandable way?
- 0 pt: Output is disorganized, unstructured, or confusing
- 2 pts: Several formatting or naming issues that hinder readability
- 5 pts: Clear, well-organized, and logically structured output

6. Examples / Analogies Quality (0.05)
- 0 pt: no example, or irrelevant/confusing example
- 2 pts: example present and relevant but only partly elaborated
- 5 pts: examples highly relevant, well-explained, and concretely aid understanding

7. Engagement & Tone (0.05)
- 0 pt: Tone includes noticeable emotional, biased, or overly casual language that detracts from educational clarity
- 2 pts: Tone is mostly appropriate but includes minor emotional or informal elements
- 5 pts: Tone is fully professional, consistent, and purely instructional with no emotional or subjective language"""