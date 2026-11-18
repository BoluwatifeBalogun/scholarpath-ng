# Results to quote in Chapter 4

Produced by `python scripts/train.py` and `python -m pytest tests -q` on the
bundled dataset. Rerun both after any change and update the numbers.

The dataset is simulated. These results show the pipeline works. They are
not evidence of accuracy on real students.

## Dataset

| | |
|---|---|
| Source | Simulated, `scripts/generate_dataset.py`, seed 4117 |
| Raw rows | 7,590 (17 columns) |
| Repeated rows dropped | 90 |
| Rows dropped for missing outcome | 22 |
| Category spellings standardised | 317 |
| Impossible values blanked | 108 |
| Blanks imputed in the pipeline | 1,007 |
| Usable rows | 7,478 |
| Coverage | 36 states and the FCT, six geopolitical zones |
| Institution types | University 3,334; Polytechnic 2,695; College of Education 1,449 |
| Records per tier (1 to 5) | 373; 1,585; 2,863; 1,611; 1,046 |

## Model

| | |
|---|---|
| Algorithm | CART Decision Tree, scikit-learn |
| Split quality measure | Entropy (information gain) |
| Deepest level allowed | 11 |
| Fewest records in a leaf | 10 |
| Chosen by | 5-fold grid search, 32 candidates, macro F1 (best CV score 0.762) |
| Split | 5,982 train / 1,496 test, stratified, seed 42 |
| Resulting tree | 627 nodes, 314 leaves, depth 11 |

## Evaluation on the 1,496 test records

| Metric | Value |
|---|---|
| Accuracy | 78.21% |
| Accuracy on training records | 86.73% (drop of 8.5 points) |
| Precision, macro / weighted | 78.60% / 78.21% |
| Recall, macro / weighted | 77.55% / 78.21% |
| F1-score, macro / weighted | 78.06% / 78.20% |

| Tier | Precision | Recall | F1 | Tested |
|---|---|---|---|---|
| 1 | 80.00% | 74.67% | 77.24% | 75 |
| 2 | 77.02% | 78.23% | 77.62% | 317 |
| 3 | 80.00% | 80.98% | 80.49% | 573 |
| 4 | 72.01% | 71.12% | 71.56% | 322 |
| 5 | 83.98% | 82.78% | 83.37% | 209 |

Confusion matrix (rows actual, columns predicted):

| | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| **T1** | 56 | 18 | 1 | 0 | 0 |
| **T2** | 14 | 248 | 55 | 0 | 0 |
| **T3** | 0 | 56 | 464 | 53 | 0 |
| **T4** | 0 | 0 | 60 | 229 | 33 |
| **T5** | 0 | 0 | 0 | 36 | 173 |

326 of 1,496 test records were misclassified, and 325 of those were placed in
a neighbouring tier. Accuracy by institution type: University 80%,
Polytechnic 74%, College of Education 82%.

The 8.5-point drop from training to test accuracy shows some overfitting.
Raising the fewest-records-in-a-leaf setting narrows it at a small cost in
test accuracy; this is worth discussing in Chapter 4.

Inputs the tree relies on most: previous GPA (54.8%), institution being a
Polytechnic (13.0%), continuous assessment (9.1%), attendance (6.7%), test
average (4.3%), library use (3.2%), outstanding carryovers (2.6%).

## System testing (Table 3.1)

| Test area | What was done | Result |
|---|---|---|
| User interface | Every page opened in Chrome at 390, 820 and 1366 px; checked for sideways overflow and script errors | Pass |
| Input validation | Empty form, text in number fields, out-of-range values, GPA above the institution's scale, wrong year for a College of Education, NaN and infinity | Pass, each rejected with a message beside the field |
| Data submission | Valid record sent through the form and through `/api/predict` | Pass |
| Model integration | Saved Joblib pipeline reloaded and compared with the one in memory | Pass, identical output |
| Prediction | Valid records return a tier, class name, class shares and the route | Pass |
| Result display | Result page, what-if panel, saved list and PDF checked | Pass |
| System integration | Register, predict, view, export; admin upload, train, evaluate, switch version | Pass |
| Access control | Signed-out access, ordinary user on admin pages, one user opening another's prediction, POST without CSRF token | Pass, all refused |

Automated suite: 56 tests, all passing.
