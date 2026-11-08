# Detection of AI Generated Misinformation

A text-based machine-learning project for identifying potentially
hallucinated or unsupported content in AI-generated responses using the
HaluEval dataset.

## Project Overview

Large language models can produce convincing responses that contain
unsupported, fabricated, or factually inconsistent information. This
project explores supervised text classification to label responses as
**supported** or **hallucinated**, alongside an evidence-comparison
component based on Natural Language Inference (NLI).

This is a decision-support tool. Predictions and NLI scores are not
guarantees that a statement is true or false.

## Objectives

-   Prepare labelled examples from HaluEval for supervised learning.
-   Convert text into numerical features using TF-IDF.
-   Train and compare Logistic Regression and Linear Support Vector
    Classification (LinearSVC).
-   Compare response-only and context-plus-response inputs.
-   Examine evidence consistency using NLI entailment, contradiction,
    and neutral scores.
-   Provide an interactive Streamlit interface.

## Dataset

The training workflow expects these HaluEval fields:

  Field                     Description
  ------------------------- ------------------------------------------
  `knowledge`               Supporting information or evidence
  `dialogue_history`        Conversation context
  `right_response`          Grounded or supported response
  `hallucinated_response`   Response containing hallucinated content

Obtain HaluEval from its official distribution and place the data where
the training script expects it. Do not upload the full dataset unless
its licence and redistribution terms permit it.

## Methods and Models

### Text preprocessing

The preprocessing pipeline checks required fields and prepares response
text and labels for feature extraction.

### TF-IDF

The explored configuration uses word-level TF-IDF with
`ngram_range=(1, 2)`, `min_df=2`, `max_features=150000`, and
`sublinear_tf=True`. It converts text into sparse numerical feature
vectors.

### Classification algorithms

-   **Logistic Regression:** a linear classifier that estimates class
    probabilities.
-   **LinearSVC:** a linear Support Vector Machine that learns a
    separating boundary between classes.

Two input configurations were evaluated: 1. **Response-only:** uses the
candidate response. 2. **Context + response:** combines knowledge,
dialogue history, and the candidate response.

### NLI-based evidence analysis

The NLI component compares supplied evidence (premise) with a statement
from the candidate response (hypothesis): - **Entailment:** the evidence
supports the statement. - **Contradiction:** the statement conflicts
with the evidence. - **Neutral:** the evidence does not establish either
relationship.

Neutral does not mean false. NLI scores depend on the quality and
completeness of the supplied evidence.

## Reported Model Results

  --------------------------------------------------------------------------------
  Input           Classifier           Test    Precision       Recall     F1-score
  configuration                    accuracy                           
  --------------- ------------ ------------ ------------ ------------ ------------
  Response-only   Logistic       **80.30%**       80.90%       79.33%   **80.11%**
                  Regression                                          

  Response-only   LinearSVC          80.20%   **81.28%**       78.47%       79.85%

  Context +       Logistic           73.43%       76.81%       67.13%       71.65%
  response        Regression                                          

  Context +       LinearSVC          74.23%       76.67%       69.67%       73.00%
  response                                                            
  --------------------------------------------------------------------------------

Among these reported experiments, **response-only Logistic Regression**
achieved the highest test accuracy and F1-score. Response-only LinearSVC
achieved the highest precision.

These values describe the reported evaluation only. The exact test-set
size and raw confusion-matrix counts have not been supplied; do not
infer them from rounded metrics.

## Project Structure

Adjust this example to match the files actually uploaded:

``` text
AI-Generated-Misinformation-Detection/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── evaluate_fact_checker.py
├── fact_checker.py
├── requirements.txt
├── README.md
├── models/
├── processed_data/
├── results/
└── tests/
```

## Requirements

-   Python 3.12 (development environment)
-   pip
-   Packages listed in `requirements.txt`

## Installation

``` bash
git clone https://github.com/<YOUR-USERNAME>/AI-Generated-Misinformation-Detection.git
cd AI-Generated-Misinformation-Detection
```

Create and activate a virtual environment.

**Windows PowerShell:**

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

After installing dependencies and placing required model files and data
in the locations expected by the code:

``` bash
streamlit run app.py
```

Open the local address shown in the terminal.

## Training and Evaluation

Check the supported arguments in your current scripts:

``` bash
python train.py --help
python evaluate.py --help
```

Use the dataset paths and options shown by these commands. Exact
command-line arguments depend on the project code.

## Limitations

-   Results depend on the coverage, quality, and labels of the HaluEval
    examples.
-   TF-IDF captures word and phrase patterns but does not fully
    represent meaning or factual truth.
-   Performance may not generalize to every domain, language, or newer
    language model.
-   NLI results depend on supplied evidence and may be inconclusive when
    evidence is missing.
-   The system supports human review; it does not replace authoritative
    fact-checking.

## Future Work

-   Evaluate on additional datasets and unseen domains.
-   Examine false-positive and false-negative examples.
-   Improve evidence retrieval for NLI analysis.
-   Calibrate classifier outputs where probability estimates are needed.
-   Expand automated tests and document reproducible experiments.

## Academic Information

-   **Project title:** Detection of AI Generated Misinformation
-   **Department:** Computer Engineering
-   **Programme:** M.Tech Data Science
-   **Institute:** COEP Technological University, Pune
-   **Project guide:** Dr. Yashodhara Haribhakta
-   **Academic year:** 2026--2027

## Acknowledgement

This project uses the HaluEval benchmark for academic work. Cite the
original HaluEval publication and follow the dataset's terms of use.

## Author

Add your name, GitHub profile, and contact information here if you want
them displayed publicly.
