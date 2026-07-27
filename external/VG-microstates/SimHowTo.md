# Running the Microstate Simulated Recovery Experiment

matlab -batch "T = simulated_ms_retrieval_experiment();"
python3 analyse_simulation_outputs.py outputs/simulations/results
python3 build_current_pipeline_manuscript.py

The first part runs in matlab using SPM (which must be on path)

Then python is used to generate the omnibus statistics and reports.
