PROMPT_TEMPLATE = """
You are generating a study guide for a student on {concept_bin}. The student provided you a file from a {context_bin}. It includes this following list of terms, definitions, and citations from the text:
   - {terms}
   - {citations}


Write a {definition_style} definition of the term, synthesizing information from the given citations.
{language_complexity}


Enhance the definition by including some of or all of the following:
   - Relevant terms that are related to the terms provided and topic
   - {example_instructions}
   - {analogy_style}
   - {practice_question}"""