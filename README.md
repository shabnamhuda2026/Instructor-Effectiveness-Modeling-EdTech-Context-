# Instructor Effectiveness Modeling (EdTech Context)

ML assignment: predicting instructor effectiveness tiers (Low / Medium / High) from batch-level EdTech data (2,000 batches, 120 instructors).

## Files
- `Accredian_assignment.ipynb`: the full notebook (EDA, effectiveness definition, aggregation, model, evaluation, interpretation, mandatory questions)
- `instructor_effectiveness_dataset_2000_rows.csv`: the dataset provided with the assignment

## Approach
1. **EDA:** distributions, correlations, and data-quality checks.
2. **Effectiveness definition:** metrics are z-scored within each course, then combined into an Instructor Effectiveness Score (Outcomes 40%, Engagement 30%, Feedback 30%). Instructor scores are shrunk toward the global mean (K = 5) and cut into percentile tiers (25/50/25).
3. **Leakage prevention:** each instructor's batches are split into two halves. One half builds the tier label and the other half builds the model features.
4. **Model:** Random Forest compared with Logistic Regression and a majority-class baseline, using 5-fold stratified cross-validation repeated 5 times.
5. **Evaluation:** classification report and confusion matrix on out-of-fold predictions.
6. **Interpretation:** feature importance, by feature and by pillar, plus answers to the five mandatory questions.

## Key results
- Baseline accuracy 0.50, Logistic Regression 0.81 ± 0.07, Random Forest 0.86 ± 0.06
- No Low↔High confusions; errors are between neighbouring tiers
- Outcomes (completion and score improvement) are the strongest drivers

## Libraries
Python, pandas, numpy, scikit-learn, matplotlib

## How to run
Place the CSV in the same folder as the notebook, then run all cells from top to bottom (Kernel → Restart Kernel and Run All Cells).
