"""
Environment (reward) model for the question generation task.

1. Generate a question from the context and answer
2. Compute the loss of the question answering model on the generated question and the initial answer
"""


from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import spacy
import benepar

from tqdm import tqdm

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer

benepar.download('benepar_en3')
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# class RewardModel(nn.Module):
#     def __init__(self, qgen_model, qans_model, tokenizer, device):
#         super(RewardModel, self).__init__()
#         self.qgen_model = qgen_model.eval().to(device)
#         self.qans_model = qans_model.eval().to(device)
#         self.tokenizer = tokenizer
#         self.prompt_qgen = self.tokenizer.encode("generate question: ", add_special_tokens=False, return_tensors="pt")
#         self.prompt_qans = self.tokenizer.encode("answer question: ", add_special_tokens=False, return_tensors="pt")
#         self.prompt_ans = self.tokenizer.encode(" answer: ", add_special_tokens=False, return_tensors="pt")
#         self.prompt_qst = self.tokenizer.encode(" question: ", add_special_tokens=False, return_tensors="pt")

#     def forward(self, qgen_batch, context_batch, answers_batch):

#         # qgen batch

        

#         # generate question given context and answer (qgen)
#         # input-> generate question: context: answer: question:
#         gen_q = self.qgen_model.generate(
#             input_ids=qgen_batch["input_ids"],
#             attention_mask=qgen_batch["attention_mask"],
#             max_new_tokens=128,
#             num_beams=8,
#             num_return_sequences=1,
#         )

#         # quesiton + padding
        

#         # answer question: question: context:

        

#         # compute loss for question answering
#         # input-> answer question: context: question:
#         # label-> [answer]

#         # outputs = self.qans_model(
#         #     input_ids=,
#         #     attention_mask=,
#         #     labels=,
#         # )
        

        
        

#         return loss, gen_q, 

    

def main():

    device = torch.device("cpu")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # question generation model
    qgen_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qg", use_auth_token=True).to(device)

    # question answering model
    qans_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qa", use_auth_token=True).to(device)

    # read data
    subsample_df = pd.read_json('psy_const_df.json')


    # For psy contexts

    # psy_df = pd.read_excel('psy.xlsx')

    # print(psy_df.head())
    
    # sent_const_list = []

    # for i, row in psy_df.iterrows():

    #     doc = nlp(row['Context'])

    #     for sent in doc.sents:

    #         const_list = list(sent._.constituents)

    #         tag_constit_list = [(constit._.labels, constit.text.strip()) for constit in const_list]

    #         sent_const_list.append({
    #             'sentence': sent.text.strip(),
    #             'spans': tag_constit_list
    #         })

    # psy_const_df = pd.DataFrame(sent_const_list)

    # psy_const_df.to_json('psy_const_df.json')

    # print(psy_const_df.head())

    # exit()


    result_list = []

    for i, row in tqdm(subsample_df.iterrows(), total=len(subsample_df)):
        sentence = row['sentence']
        span_list = [x for x in row['spans'] if x[0] != []]

        filtered_span_list = []

        for span in span_list:
            # eliminate spans with more than 10 words
            if len(span[1].split()) > 8 and span[0][0].startswith('WH'):
                continue

            filtered_span_list.append(span[1])

        # append contexts and prompts

        qgen_list = [f"generate question: answer: {x} context: {sentence}  "  for x in filtered_span_list]


        # create batch for question generation
        
        qgen_batch = tokenizer.batch_encode_plus(qgen_list, max_length=1024, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)

        # generate questions
        gen_q = qgen_model.generate(
            input_ids=qgen_batch["input_ids"],
            attention_mask=qgen_batch["attention_mask"],
            max_new_tokens=64,
            num_beams=8,
            num_return_sequences=1,
        )

        # decode questions
        gen_q = tokenizer.batch_decode(gen_q, skip_special_tokens=True)

        # print questions
        for q in gen_q:
            print(q)

        qans_batch = tokenizer.batch_encode_plus([f"answer question: {x} context: {sentence}  " for x in gen_q], max_length=1024, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        ans_batch = tokenizer.batch_encode_plus(filtered_span_list, max_length=128, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)


        # compute loss for question answering
        # input-> answer question: context: question:
        # label-> [answer]

        outputs = qans_model(
            input_ids=qans_batch["input_ids"],
            attention_mask=qans_batch["attention_mask"],
            labels=ans_batch["input_ids"],
        )
        
        # compute loss for each question
        loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

        # compute loss for each question
        loss_list = []
        for i in range(len(gen_q)):
            loss_list.append(loss(outputs["logits"][i], ans_batch["input_ids"][i]))

        result_list = []
        for i in range(len(gen_q)):
            # print("Context: ", sentence)
            # print("Question: ", gen_q[i])
            # print("Answer: ", filtered_span_list[i])
            # print("Loss: ", loss_list[i])
            # print("+++++++++")

            result_list.append({
                "context": sentence,
                "question": gen_q[i],
                "answer": filtered_span_list[i],
                "loss": loss_list[i].item()
            })

    # create dataframe
    result_df = pd.DataFrame(result_list)

    # save dataframe to xlsx
    result_df.to_excel("result_psy.xlsx")


if __name__ == "__main__":
    main()