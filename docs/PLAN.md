# ScholarPath NG: algorithm, system flow and build notes

## 1. What the report commits the system to

- A web application: Flask backend, HTML/CSS/JavaScript front-end.
- A Decision Tree Classifier built with scikit-learn, data handled with Pandas,
  the trained model saved with Joblib, SQLite for stored information.
- Two actors (Figure 3.3): the Student / Academic Officer, who enters and
  submits information and views the prediction; and the System Administrator,
  who uploads and manages the dataset, prepares it, trains the tree and
  evaluates it.
- Three layers (Figure 3.2): presentation, processing, model.
- Evaluation with accuracy, precision, recall, F1-score and a confusion matrix
  on records not used in training (Section 3.6).
- Unit, integration, system, input validation and prediction testing
  (Section 3.8, Table 3.1).
- The system supports decisions. It does not replace lecturers or grading.

## 2. Offline process: from dataset to saved model

```
Raw CSV
  -> 1 inspect            rows, columns, types, missing cells, duplicates
  -> 2 clean              consistent category spellings
                          numbers forced to numeric
                          duplicate rows removed
                          rows with no outcome removed
                          out-of-range values set to missing
  -> 3 select features    12 inputs kept; id, gender, state, zone dropped
                          final CGPA kept apart as the outcome
  -> 4 derive target      tier 1 to 5 from final CGPA and the institution's
                          grading bands (scholarpath/grading.py)
  -> 5 split              stratified 80 / 20, fixed seed
  -> 6 fit pipeline       on the training part only:
                          previous GPA -> percent of the institution's scale
                          median imputation (numbers)
                          most-frequent imputation + ordinal encoding
                          one-hot encoding of institution type
                          DecisionTreeClassifier (CART)
  -> 7 evaluate           on the untouched test part
  -> 8 save               whole pipeline with Joblib, stored as a model version
  -> 9 activate           the administrator makes a version the active model
```

No feature scaling is applied, as Section 3.4.4 says: a tree splits on
thresholds, so scale does not matter.

### The Decision Tree

CART as implemented in scikit-learn. At each node it tries every feature and
every threshold and keeps the split with the largest drop in impurity:

- Gini impurity: `1 - sum(p_k^2)`
- Entropy: `-sum(p_k * log2(p_k))`; information gain is the drop in entropy

Splitting stops when a node is pure, when the maximum depth is reached, or
when a split would leave a leaf with fewer than the minimum number of records.
The last two are the guards against overfitting from Section 2.5.3.

Default settings (entropy, depth 11, at least 10 records per leaf) were chosen
by a 5-fold cross-validated grid search over 32 combinations scored on macro
F1. The administrator can rerun that search or set values by hand.

### Inputs

| Input | Type | Range |
|---|---|---|
| Institution type | category | University, Polytechnic, College of Education |
| Year of study | whole number | 1 to 4 (1 to 3 for Colleges of Education) |
| UTME score | whole number | 100 to 400 |
| Previous GPA | number | 0 to the institution's scale (4 or 5) |
| Continuous assessment | number | 0 to 30 |
| Test average | number | 0 to 100% |
| Outstanding carryovers | whole number | 0 to 12 |
| Lecture attendance | number | 0 to 100% |
| Private study | number | 0 to 60 hours a week |
| Assignments submitted | number | 0 to 100% |
| Library use | number | 0 to 40 hours a week |
| Access to learning resources | ordered category | Low, Medium, High |

Output: a performance tier, 1 (highest) to 5, named according to the
institution type.

## 3. Online process: one prediction (Figure 3.4)

```
START
 -> open the application, sign in
 -> enter student information            templates/predict.html
 -> validate input                        schema.validate_input
      invalid -> show the message beside each field, return to the form
 -> prepare input                         preprocessing.to_frame + saved pipeline
 -> load the trained classifier           services.load_pipeline
 -> generate the prediction               modeling.predict
 -> display the predicted category        templates/result.html
END
```

The result also carries the share of training records in the same leaf that
had each class, the exact tests the record passed from root to leaf, and
improvement recommendations. The recommendations are fixed rules on the
entered values (`scholarpath/recommend.py`), stated as such on screen; the tree
does not produce them. Each prediction can be exported as a PDF.

Two extra routes reuse the same validate, prepare and classify steps without
saving anything: `/api/demo` behind the home-page demo (three inputs vary,
the other nine are held at typical values) and `/api/whatif` behind the
what-if panel on a result.

## 4. Storage

| Table | Holds |
|---|---|
| users | accounts, hashed passwords, role |
| datasets | the uploaded CSV and its inspection and cleaning report |
| model_versions | settings, full evaluation report, the Joblib file, active flag |
| predictions | inputs, predicted tier and label, decision path, who and when |

## 5. Security and data care

- Passwords are hashed (Werkzeug scrypt). Sessions are signed cookies.
- Every form carries a CSRF token.
- Users see only their own predictions. Admin pages return 403 to others.
- Gender and location are never model inputs.
- The student reference is optional and is not used by the model.

## 6. Front-end

- Theme: the colours of the Nigerian coat of arms. Black shield, white, green,
  the yellow of Costus spectabilis, and the eagle's red kept for the lowest
  tier and for errors. Display type Archivo (expanded), text type Public Sans,
  both self-hosted so the app works offline at a defence.
- The mark is the white Y from the shield, the meeting of the Niger and the
  Benue, which is also the shape of a single tree split.
- Signature element: a live demo in the hero. Three sliders call the trained
  model and a marker moves along the five-tier scale.
- Layout: a public site with a top bar, and a signed-in app with a fixed side
  menu on desktop and a bottom dock on phones.
- From the ui-ux-pro-max skill: 4.5:1 text contrast, 44px touch targets,
  visible focus, labels above inputs, errors beside the field plus a linked
  summary, validation on blur, reduced-motion support, SVG icons only, checks
  at 390, 820 and 1366 px.
- Interaction patterns rebuilt in plain CSS and JavaScript: hard-shadow press
  buttons, live sliders, a scroll progress bar, numbers that count up, steps
  that light in turn, a route that fills in row by row.
- Charts are plain HTML and CSS: one colour for magnitude, the count printed
  in every confusion-matrix cell, a table beside each chart, tier colours
  always paired with the tier name.

## 7. Things worth fixing in the report itself

- Figure numbering: the List of Figures gives the system model as 3.2 and the
  use case as 3.3, but the caption under the use case reads "Figure 3.2", and
  the integration workflow is captioned "Figure 3.4" (the same number as the
  flowchart) where the list has 3.5.
- Sections 3.2 and 3.2.1 are titled "Research Design" and "Research Process"
  but describe the existing system and its limitations.
- Sections 3.4 and 3.9 say the dataset will be documented after selection.
  Chapter 4 must state that the data is simulated, or name the real source
  if the group obtains one.
