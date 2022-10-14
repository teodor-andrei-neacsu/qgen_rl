# Reinforcement Learning for Question Generation

Description:

The goal of the agent is to:
1. select a span of text that will represent the answer
2. select a span of text related to that answer that will represent the question 
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

Description:

A - action
R - reward
E - environment
A - agent

The evaluation architecture:

How to evaluate a question






