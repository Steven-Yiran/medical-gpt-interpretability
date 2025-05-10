"""
This module contains prompts for medical question answering.
"""

# see meerkat huggingface model repo for more details
meerkat_medqa_system_prompt_cot = "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"
meerkat_medqa_system_prompt_direct = "The following is a multiple-choice question about medical knowledge. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; state your response with the phrase \"the answer is ([option_id]) [answer_string]\".\n\n"

mistral_medqa_system_prompt = "The following is a multiple-choice question about medical knowledge. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"Answer: ([option_id])\".\n\n"

# Bare evaluation prompt template
prompt_eval_bare = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D}"""

prompt_eval_with_bracket = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D} The answer is ("""

# Bare evaluation prompt template
prompt_eval_bare_mistral = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D} Answer: ("""

prompt_eval_patient_mistral = """{question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D} \nSummary: The patient is a"""

# You can add more prompt templates here as needed
prompt_eval_cot = """The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase "the answer is ([option_id]) [answer_string]".

USER: {question} (A) {choice_A} (B) {choice_B} (C) {choice_C} (D) {choice_D} ASSISTANT: """