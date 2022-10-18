"""
Environment (reward) model for the question generation task.


Propagate the batch of samples (context, answer) to generate the questions. => question
Propagate the batch of samples (context, question) to generate the answers. => answer'

Compute the reward considering:
    - the similarity between the answer and the answer'
    - the quality of the question (might need a model)


"""


import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, qgen_model, qans_model, tokenizer):
        super(RewardModel, self).__init__()
        self.qgen_model = qgen_model.eval()
        self.qans_model = qans_model.eval()
        self.tokenizer = tokenizer

    def generate_questions(self, context, answer):
        # generate the questions
        pass

    def generate_answers(self, context, question):
        # generate the answers
        pass

    def compute_reward(self, answer, answer_prime):
        # compute the reward
        pass

    def compute_similarity(self, answer, answer_prime):
        # compute the similarity
        pass

    def compute_question_quality(self, question):
        # compute the question quality
        pass

    def forward(self, context, answer, answer_prime):
        # compute the reward
        pass

    

def main():

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # question generation model
    qgen_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qg")

    # question answering model
    qans_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qa")

    # create the reward model
    reward_model = RewardModel(qgen_model, qans_model, tokenizer)


if __name__ == "__main__":
    main()