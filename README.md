# Reinforcement Learning for Question Generation

Description:
Train a reinforcement learning agent to generate questions. SQuAD dataset will be initially used for training.

Stage 0: (Extractive) Single sentence question generation with asnwer span.
Stage 1: (Extractive) Single sentence question generation. Both spans are in the same sentence.
Stege 2: (Extractive) Multi sentence question generation. Spans can be in different sentences.
Stage 3: Multi sentence question generation. In this stage we need an entailment/reasoning model to generate questions.

The goal of the agent is to:
1. select a span of text that will represent the answer
2. select a span of text (related) to that answer that will represent the question 
    + what does related in this case mean?
3. generate a question based on the selected spans
4. obtain a reward based on the quality of the question (this will include the diversity of the questions | divergent questions generate higher rewards)
5. repeat from step 2 until the agent selected best questions for the selected answer

The architecture has 2 main components:

The reinforcement learning agent:

The state of the agent will have the following information:
1. the answer span
2. first question span
3. second question span
4. ...

S = (span_A, span_Q1, span_Q2, ...)

The agent must learn also when it has exhausted all the possible question spans => when to stop


Dataset:
+ SQuAD 

Model:
1. T5 for question answering
2. T5 for question generation


TODO:
1. For each senetence generate multi-granularity spans (word, subword, sentence) based on constituency tree