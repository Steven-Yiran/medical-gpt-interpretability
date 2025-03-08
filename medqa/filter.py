import json
import re
import sys
import unicodedata

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
        match = regex.findall(resp)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict: 
                match = convert_dict[match]
        return match


    def extract_answer(self, response, choices=None):
        # give tally of each cases of match
        matchFirst = re.search(r'the answer is .(\w).', response)
        if matchFirst:
            return f"({matchFirst.group(1)})"
        # also find "the correct answer is"
        matchSecond = re.search(r'the correct answer is .(\w).', response)
        if matchSecond:
            return f"({matchSecond.group(1)})"
        match = self.find_match(self.regex, response) 
        if match:
            return f"({match})"
        return "[invalid]"

    def filter_responses(self, responses, choices):
        return [self.extract_answer(resp, choices) for resp in responses]


def main():
    #model_name = "KrithikV/MedMobile"
    #model_name = model_name.split("/")[-1]
    path = "meerkat-7b-v1.0_results.json"
    #path = "Llama-3.1-8B-Instruct_results.json"
    with open(path, "r") as f:
        results = json.load(f)

    correct = 0
    incorrect = 0

    filter = MultiChoiceFilter()
    choice_tally = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for result in results:
        generated_response = result["generated_response"]
        answer = filter.extract_answer(generated_response)
        gold = result["gold"]
        answer = answer.replace("(", "").replace(")", "")

        if answer not in ['A', 'B', 'C', 'D']:
            incorrect += 1
        else:
            choice_tally[answer] += 1
            answer_num = ord(answer) - ord('A')
            if answer_num == gold:
                correct += 1
            else:
                incorrect += 1

    print(f"Correct: {correct}, Incorrect: {incorrect}")
    print(f"Accuracy: {correct / (correct + incorrect)}")
    print(f"Choice Tally: {choice_tally}")
if __name__ == "__main__":
    main()