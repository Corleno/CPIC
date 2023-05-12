# Compressed Predictive Information Coding 
This repo is used to publish the code for Compressed Predictive Information Coding (CPIC) methodology

## Preinstall Pacakages
CPIC requests the pre-installation of DCA package for initialization. Please refer to https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/index.html.

## Code

### Main Code
1. Main code:
   1. CPIC code: <code> code/CPIC.py </code>
   2. utility code: <code> utils/* </code>

### Synthetic Experiments

1. Lorenz experiment code: <code> code/synthetic_*.py</code>
   1. Synthetic data generation using <code> synthetic_generation.py</code>
   2. CPIC model on synthetic data using <code> synthetic_experiment.py</code>  
   3. Other models on synthetic data using <code> code/synthetic_competitors.py</code>
   4. Per/Post analysis includes <code>synthetic_plot.py, synthetic_summarization.py, synthetic_summary.py, synthetic_visualization.py</code>
2. Synthetic experiments to understand the Prediction information in CPIC setting: <code> code/synthetic/*</code>
   1. Synthetic data generation with <code> code/synthetic/data_generator.py </code>
   2. PI analysis using <code> code/synthetic/PI_analysis.py </code>


### Real Data Experiments
1. Real data experiments code: <code> code/real_data_*.py </code>
   1. Real data experiments with CPIC for four datasets including M1, HC, Temp, MS using <code> real_data_experiment_standard.py, real_data_experiment_standard_beta.py </code>. Note that beta refers to varying weight option.
   2. Real data experiments with other models using <code> real_data_competitors.py </code>
   3. Real data experiments post analysis: <code> real_data_summary_standard.py, real_data_summary_standard_beta.py </code>

## Configuration
All configurations for CPIC are available in <code> config/* </code>

## Figures
All figures in the paper are available in <code> fig/* </code>

