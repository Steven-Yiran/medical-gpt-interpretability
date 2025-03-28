"""
This module contains prompts for medical question answering.
"""

meerkat_medqa_system_prompt = "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"

# Bare evaluation prompt template
prompt_eval_bare = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D}"""

# You can add more prompt templates here as needed
prompt_eval_cot = """The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase "the answer is ([option_id]) [answer_string]".

USER: {question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D} ASSISTANT: """