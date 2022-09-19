The file expt.csv includes data for the experiment presented in Section 4.1. THe file expt_old.csv includes a previous version of the experiment presented in Appendix B.

The fields in the CSV data are described as follows:

- `progress` and `finished`: The progress of completion.

If the progress is 100%, the survey is finished. In practice, the survey can be unfinished, because the worker does not pass either of the two trials of the attention check questions, or the worker decides to abandon the survey, or

In our data analysis, we consider all workers who have provided complete numerical answers to the items they evaluate (5 items for the 5Q-group, and 20 items for the 20Q-group.)

- `attention_{1,2,3}`: Worker responses to the three attention check questions.

The correct answers are {2, 1, 2} respectively.

- `attention_[1,2,3]_fail`: worker

If the worker fails at least one of the attention check questions incorrectly, they are given a second chance to answer the three questions again.

- `q{1-20}g20` and `q{16-20g}g5`: The final answers given by the workers, for the 20 questions in the 20Q-group, and the 5 questions in the 5Q-group.

- `method_description`: The workers' free-form description of the strategy they use to answer questions.

- `time_q{1-20}`: The timestamps associated with initial answer and subsequent modifications for each question. The 5Q-group only has timestamps for Q16-20.

- `history_q{1-20}`: The sequence of answers (initial answer and subsequent modifications) for each question. The 5Q-group only has answers for Q16-20.

- `button_{k+1}_to_{k}` (k=1, 2, 3): The number of times that the button for turning from page (k+1) to page k is clicked.

- `time_button_{k+1}_to_{k}` (k=1, 2, 3) [new experiment only]: The timestamps associated with each button click for turning from page (k+1) to page k.

- `setting`: The setting for generating true values.
