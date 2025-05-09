import json
import re
import sys
import unicodedata
import random
import torch
from tqdm import tqdm

class MultiChoiceFilter:
    # Inspiring from lmeval
    def __init__(self, ignore_case=False, ignore_punctuation=False, regex_pattern=r"[\(\[]([A-D])[\)\]]"):
        
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) 
                                       if unicodedata.category(chr(i)).startswith("P"))

    def filter_text(self, text):
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(self.punct_tbl)
        return text

    def find_match(self, regex, resp, convert_dict={}):
        # look for the regex patter from the end to the start
        # Search from the end to the start by reversing the string
        reversed_resp = resp[::-1]
        match = regex.search(reversed_resp)
        if match:
            # Extract the match and reverse it back to get the correct order
            match = match.group(0)[::-1]
            # Extract the actual letter from the match using the original regex
            match = regex.search(match)
            if match:
                match = match.group(1)
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
        return match

    def extract_answer(self, response, choices=None):
        match = re.search(r'(?:therefore,\s*)?the answer is.*?\(([A-D])\)', response, re.IGNORECASE)
        if match:
            return match.group(1), "(therefore), the answer is <ans>"

        match = re.search(r'answer:\s*\(([A-D])\)', response, re.IGNORECASE)
        if match:
            return match.group(1), "answer: (ans)"

        match = re.search(r'answer:\s*([A-D])\)', response, re.IGNORECASE)
        if match:
            return match.group(1), "answer: ans)"

        match = re.search(r'answer:\s*\n\(([A-E])\)', response, re.IGNORECASE)
        if match:
            return match.group(1), "answer: [new line] (ans)"

        match = re.search(r'the answer is.*?\[([A-D])\]', response, re.IGNORECASE)
        if match:
            return match.group(1), "the answer is"

        match = re.search(r'answer is .(\w).', response)
        if match:
            return match.group(1), "answer is"
        match = re.search(r'answer is.*?\[([A-D])\]', response, re.IGNORECASE)
        if match:
            return match.group(1), "answer is"
        match = self.find_match(self.regex, response) 
        if match:
            return match, "single letter"
        return "invalid", "invalid"

    def filter_responses(self, responses, choices):
        return [self.extract_answer(resp, choices) for resp in responses]
    
class PatientInfoFilter:
    def __init__(self):
        # The previous regex had issues matching common age patterns
        # This new pattern will match formats like:
        # "A 40-year-old man", "A 40 year old woman", "An 85-yr-old patient"
        self.regex = re.compile(r'a \d+[-\s]year[-\s]old (?:man|woman)', re.IGNORECASE)
        self.gender_keywords = {
            "man": "male",
            "woman": "female",
            "male": "male",
            "female": "female",
            "boy": "male",
            "girl": "female",
        }
        self.pronoun_map = {
            "male": {"he": "she", "his": "her", "him": "her"},
            "female": {"she": "he", "her": "his", "hers": "his"},
        }

    def filter_text(self, text):
        return self.regex.search(text)
    
    def extract_gender(self, question):
        # Use regex to find gender-related keywords
        pattern = r'\b(?:' + '|'.join(self.gender_keywords.keys()) + r')\b'
        match = re.search(pattern, question.lower())
        if match:
            return self.gender_keywords[match.group().lower()]
            
        # If no direct gender keyword found, look for gender-specific pronouns
        he_pattern = r'\b(?:he|his|him)\b'
        she_pattern = r'\b(?:she|her|hers)\b'
        
        he_matches = len(re.findall(he_pattern, question.lower()))
        she_matches = len(re.findall(she_pattern, question.lower()))
        
        if he_matches > she_matches:
            return "male"
        elif she_matches > he_matches:
            return "female"
            
        return None  # Return None if no gender indicators found
    

class GenderConditionFilter:
    """
    Filter out responses that contain gender-specific medical conditions such as breast cancer, prostate cancer, etc.
    """
    def __init__(self):
        male_conditions = [
            "prostate",
            "testicular",
            "testicular",
            "erectile",
            "hypogonadism",
            "varicocele",
            "priapism",
            "phimosis"
        ]
        female_conditions = [
            "breast",
            "ovarian",
            "cervical",
            "endometriosis",
            "vaginitis",
            "vulvodynia",
            "menstrual",
            "mastitis",
            "pregnancy",
        ]
        gender_terms = [
            "pregnancy",
            "menstruation",
            "ovulation",
            "semen",
            "sperm",
            "prostate",
            "vagina",
            "uterus",
            "ovaries",
            "menopause",
            "breastfeeding",
            "labor",
            "delivery"
        ]
        self.condition_keywords = male_conditions + female_conditions + gender_terms
        self.count = 0

    def filter_text(self, text):
        if any(keyword in text.lower() for keyword in self.condition_keywords):
            self.count += 1
            return True
        return False

def find_assistant_response(prompt, pattern="ASSISTANT:"):
    # find the last instance of the pattern
    last_instance = prompt.rfind(pattern)
    if last_instance == -1:
        return None
    return prompt[last_instance + len(pattern):]

def truncate_answer_text(prompt, pattern):
    # truncate the prompt until the end of the pattern "the answer is ("
    # since there are multiple instances of this pattern in the prompt, we need to find the last one
    # find the last instance of the pattern
    last_instance = prompt.rfind(pattern)
    if last_instance == -1:
        return None
    return prompt[:last_instance + len(pattern)]

def setup_prompt(prompt, model_name):
    if "mistral" in model_name.lower():
        prompt = truncate_answer_text(prompt, "Answer: (")
        prompt = find_assistant_response(prompt, "Answer: ([option_id])\".\n\n")
    else:
        prompt = truncate_answer_text(prompt, "the answer is (")
        prompt = find_assistant_response(prompt)
    return prompt

def generate_counterfactual_patient_info(prompt, patient_gender, swap_gender=False, swap_pronouns=False):
    """
    Generate a counterfactual version of patient information by changing gender.
    
    Args:
        prompt (str): The original prompt containing patient information
        
    Returns:
        str: A counterfactual version of the prompt with gender changed
    """
    pronoun_map = {
        "male": {"he": "she", "his": "her", "him": "her", "He": "She", "His": "Her", "Him": "Her"},
        "female": {"she": "he", "her": "his", "hers": "his", "She": "He", "Her": "His", "Hers": "His"},
    }
    if swap_gender:
        if patient_gender == "male":
            prompt = re.sub(r"\bman\b", "woman", prompt)
            prompt = re.sub(r"\bmale\b", "female", prompt)
        else:
            prompt = re.sub(r"\bwoman\b", "man", prompt)
            prompt = re.sub(r"\bfemale\b", "male", prompt)
    if swap_pronouns:
        replace_map = pronoun_map[patient_gender]
        for old, new in replace_map.items():
            prompt = re.sub(r'\b' + re.escape(old) + r'\b', new, prompt)
    return prompt

def select_random_answer(correct_answer):
    wrong_answers = ['A', 'B', 'C', 'D']
    wrong_answers.remove(correct_answer)
    return random.choice(wrong_answers)

def assert_logits_match(model, hf_model, tokenizer):
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ", 
        "My favorite", 
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]

    model.eval()
    hf_model.eval()
    prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]

    tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]
    logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

    for i in range(len(prompts)):
        assert torch.allclose(logits[i], tl_logits[i], atol=1, rtol=1e-1)