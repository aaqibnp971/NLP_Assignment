# NLP Hallucination Detection Pipeline

This repository contains the codebase, notebooks, and deliverables for the NLP Assignment (CS F429) focused on **Hallucination Detection in Natural Language Processing**.

## 📂 Project Structure

The project is structured into a sequence of Jupyter Notebooks that document the end-to-end machine learning pipeline:

- **`01_data_exploration.ipynb`**: Loading and preprocessing datasets (`HaluEval` and `RAGTruth`), analyzing label distributions, and data cleaning.
- **`02_core_pipeline.ipynb`**: Implementation of the core modeling pipeline and feature engineering.
- **`03_baselines.ipynb`**: Establishing baseline model performance for hallucination detection.
- **`04_experiments.ipynb`**: Advanced experimentation, hyperparameter tuning, and detailed evaluation metrics.
- **`05_demo .ipynb`**: A demonstration notebook showcasing the final model's capabilities on custom text inputs.

## 📄 Deliverables
The repository also includes the final project reports and presentations submitted by **Team 8**:
- `NLP_Project_Report.pdf` / `NLP_Report_Team_8.docx`
- `NLP_PPT_TEAM_8.pptx`
- `Presentation_Script_NLP.pdf`
- `Team_Prep_Guide_NLP.pdf`

## 🚀 Setup & Installation

To run this project locally, it is recommended to use a virtual environment.

1. Clone this repository:
   ```bash
   git clone https://github.com/aaqibnp971/NLP_Assignment.git
   cd NLP_Assignment
   ```
2. Install the necessary dependencies (ensure you have Jupyter installed):
   ```bash
   pip install datasets pandas scikit-learn jupyterlab
   ```
   *(Note: Adjust the installed packages based on your specific environment requirements).*

3. Run Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter lab
   ```

## ⚠️ Notes on Data
Due to GitHub's 100MB file size limits, large datasets (such as `ragtruth.csv`) are not tracked in this repository. The `01_data_exploration.ipynb` notebook contains the code to pull and process the dataset directly from the Hugging Face Hub (`wandb/RAGTruth-processed`).


