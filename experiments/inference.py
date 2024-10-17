import argparse

from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Generate text using BioGPT")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # Load prompt from the user, cannot be empty
    aa("--interactive", action="store_true", help="Whether to run in interactive mode")
    args = parser.parse_args()

    print("Running in interactive mode")
    if args.interactive:
        while True:
            question = input(": ")
            if not question:
                print("Prompt cannot be empty")
                continue
            prompt = (
                f"Question: {question} the answer to the question is"
            )
            generate_text(prompt)
            

def generate_text(prompt):
    model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    # add gpu device
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cuda:0")
    set_seed(42)
    result = generator(
        prompt,
        max_length=40,
        truncation=True,
    )

    print(result[0]["generated_text"][len(prompt):])

if __name__ == "__main__":
    main()