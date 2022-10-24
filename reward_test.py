
from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer


device = torch.device("cpu")

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# question generation model
qgen_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qg", use_auth_token=True).to(device)

# # question answering model
qans_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qa", use_auth_token=True).to(device)


# # create the reward model
# reward_model = RewardModel(qgen_model, qans_model, tokenizer).to(device)

subsample_df = pd.read_json('squad_df_validation.json')

print(subsample_df.head())

train_sample = subsample_df.iloc[1]
train_context = train_sample['sentence']
train_spans = train_sample['spans']

# the input to the model:
# for question generation: generate question: answer: context:
# for question answering: answer question: context, question

train_ans = train_spans[10][1]

print("Train context: ", train_context)
print("Train ans:     ", train_ans)

"""
input: context, answer
qgen input: generate question: answer: context:
prompt_tokens = tokenizer.encode("generate question: ", add_special_tokens=False, return_tensors="pt")



"""


# tokenize input for question generation 
input_id_a = tokenizer("generate question: answer: " + train_ans + " context: " + train_context, return_tensors="pt", max_length=128, padding='max_length', truncation=True)

# print("Input ids: ", input_id_a)

train_ans_tok = tokenizer(train_ans, return_tensors="pt")


input_decoded = tokenizer.decode(input_id_a['input_ids'][0], skip_special_tokens=True)
# print("Input decoded: ", input_decoded)

# generate question
gen_q = qgen_model.generate(input_ids=input_id_a["input_ids"].to(device),
                            attention_mask=input_id_a["attention_mask"].to(device),
                            max_new_tokens=64,
                            num_beams=8,
                            num_return_sequences=1,
                            )

print("Qgen Outputs: ", gen_q[0])
print("Qgen Outputs decoded: ", tokenizer.decode(gen_q[0], skip_special_tokens=False))

# # decode the question
question = tokenizer.decode(gen_q[0], skip_special_tokens=True)

input_qans = tokenizer(f"answer question: question: {question} context: {train_context}", return_tensors="pt", max_length=128, padding='max_length', truncation=True)

sel_tok_ans = train_ans_tok["input_ids"]
sel_tok_att = train_ans_tok["attention_mask"]

print("Ans: ", train_ans)
# print("Sel tok ans: ", sel_tok_ans)
# print("Sel tok att: ", sel_tok_att)

outputs = qans_model(
    input_ids=input_qans["input_ids"].to(device),
    attention_mask=input_qans["attention_mask"].to(device),
    labels=sel_tok_ans.to(device),
)

# print("Qans Outputs: ", outputs)
print("loss: ", outputs.loss)
print("Ans shape", sel_tok_ans.shape)
print("Qans Outputs: ", outputs['logits'].shape)

# argmax the logits
pred_ans = torch.argmax(outputs['logits'], dim=2)
# decode the answer

answer = tokenizer.decode(pred_ans[0], skip_special_tokens=False)
print("Decoded pred answer: ", answer)