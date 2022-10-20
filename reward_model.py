"""
Environment (reward) model for the question generation task.


Propagate the batch of samples (context, answer) to generate the questions. => question
Propagate the batch of samples (context, question) to generate the answers. => answer'

Compute the reward considering:
    - the similarity between the answer and the answer'
    - the quality of the question (might need a model)


"""


from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, qgen_model, qans_model, tokenizer):
        super(RewardModel, self).__init__()
        self.qgen_model = qgen_model.eval()
        self.qans_model = qans_model.eval()
        self.tokenizer = tokenizer

    def generate_questions(self, context, answer):
        """
        Generate the questions from the context and the answer.

        Args:
            context (): the context
            answer (str): the answer
        
        """
        pass

    def generate_answers(self, context, question):
        # generate the answers
        pass

    def forward(self, context, answer):
        

        question = self.generate_questions(context, answer)
        


        pass

    

def main():

    device = torch.device("cpu")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # question generation model
    qgen_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qg", use_auth_token=True).to(device)

    # # question answering model
    qans_model = T5ForConditionalGeneration.from_pretrained("readerbench/t5-qa", use_auth_token=True).to(device)

    # # create the reward model
    # reward_model = RewardModel(qgen_model, qans_model, tokenizer).to('mps')

    subsample_df = pd.read_json('squad_df_train_subsample.json')
    train_sample = subsample_df.iloc[0]
    train_context = train_sample['sentence']
    train_spans = train_sample['spans']

    # the input to the model:
    # for question generation: generate question: answer: context:
    # for question answering: answer question: context, question

    train_ans = train_spans[0][1]

    print("Train context: ", train_context)
    print("Train ans:     ", train_ans)
 

    # tokenize input for question generation 
    input_id_a = tokenizer("generate question: answer: " + train_ans + " context: " + train_context + " question: ", return_tensors="pt", max_length=128, padding='max_length', truncation=True)

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

    input_qans = tokenizer(f"answer question: question: {question} context: {train_context} answer: ", return_tensors="pt", max_length=128, padding='max_length', truncation=True)

    # # print("Input ids:     ", input_qans)

    # input_decoded = tokenizer.decode(input_qans['input_ids'][0], skip_special_tokens=True)

    # # print("Input decoded: ", input_decoded)

    # # generate answer

    # outputs = qans_model.generate(input_ids=input_qans["input_ids"].to(device),
    #                             attention_mask=input_qans["attention_mask"].to(device),
    #                             max_new_tokens=128,
    #                             num_beams=8,
    #                             num_return_sequences=1,
    #                             output_scores=True,
    #                             return_dict_in_generate=True
    #                             )

    # print("Qans Outputs: ", outputs)

    # # decode the answer
    # answer = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)

    # print("Answer:      ", answer)


    sel_tok_ans = train_ans_tok["input_ids"]
    sel_tok_att = train_ans_tok["attention_mask"]

    # print("Decoded sel_tok_ans: ", tokenizer.decode(sel_tok_ans[0], skip_special_tokens=False))
    # print("Decoded input_qans: ", tokenizer.decode(input_qans['input_ids'][0], skip_special_tokens=False))

    print("Ans: ", train_ans)
    print("Sel tok ans: ", sel_tok_ans)
    print("Sel tok att: ", sel_tok_att)

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

    print("Pred ans: ", pred_ans)


    answer = tokenizer.decode(pred_ans[0], skip_special_tokens=False)
    print("Decoded answer: ", answer)

    # outputs = qans_model.generate(input_ids=input_qans["input_ids"].to(device),
    #                             attention_mask=input_qans["attention_mask"].to(device),
    #                             decoder_input_ids=sel_tok_ans.to(device),
    #                             decoder_attention_mask=sel_tok_att.to(device),
    #                             max_new_tokens=128,
    #                             num_beams=8,
    #                             num_return_sequences=1,
    #                             output_scores=True,
    #                             return_dict_in_generate=True)

    # print("Qans (sel) Outputs: ", outputs["sequences"][0])
    # print("Decoder output: ", tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True))









if __name__ == "__main__":
    main()