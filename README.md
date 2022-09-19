# Allocation Schemes in Analytic Evaluation: Applicant-Centric Holistic or Attribute-Centric Segmented

This is the code and data for the paper "Allocation Schemes in Analytic Evaluation: Applicant-Centric Holistic or Attribute-Centric Segmented?".

The participants are recruited on the Prolific crowdsourcing platform, and are required to be based in the US. The experiments conducted are approved by the IRB at CMU.

Additional de-identified meta-data collected for the experiments (including participants click and timing information, and free-form text responses) is available upon request.

## Code for simulation
The directory `simulation` includes the code for replicating the simulation results.

To replicate the experimental results reported in the paper (in Section 3), run the command (for Section 4.1, 4.2 and 4.3 respectively):
```
python simulation_calibration.py
python simulation_efficiency.py
python simulation_bias.py
```

## Experiments

#### Interface
The directory `interface` includes the user interface for the crowdsourcing experiments. `interface.pdf` includes the interface when the worker answers all 3 attention questions correctly. `interface_attn_incorrect.pdf` includes the interface when the worker answers at least 1 attention question incorrectly. Then the worker is given a second chance to answer the same 3 attention questions. If the worker answers all 3 attention questions correctly in the second time, the worker resumes with the normal experiment (Page 6 in `interface.pdf`).

#### Data generation
The directory `experiments` includes the crowdsourcing data and code for the experiment.

`generate_randomness_qualtrics.py` generates the random numbers presented to the workers.

See `experiments/data/README.md` for detailed description of the crowdsourcing data collected.

#### Data analysis
`analysis.py` performs data analysis.

## Citation
```
@article{wang2022allocation,
	author    = {Jingyan Wang and Carmel Baharav and Nihar B. Shah and Anita Williams Woolley and R Ravi},
  title	    = {Allocation schemes in analytic evaluation: {A}pplicant-centric holistic or attribute-centric segmented?},
	journal   = {Proceedings of the AAAI Conference on Human Computation and Crowdsourcing (HCOMP)},
	year      = {2022},
}
```

## Contact
If you have any questions or feedback about the data, code or the paper, please contact Jingyan Wang (jingyanw@gatech.edu).
