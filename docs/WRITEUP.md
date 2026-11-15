# HeartPost: heart-disease triage for rural clinics without a cardiologist

**Track:** Health · **Team:** Yeva (captain), Ipatiy, Ariana

> **This project is a prototype and is not medical advice.**

**Live tool:** https://avenyx-alt.github.io/HeartPost/ · **Code:** https://github.com/Avenyx-alt/HeartPost · **Video:** https://youtu.be/onxwtUJJg8Q

## The problem
46% of US counties have no practicing cardiologist, including 86% of rural counties. Patients there drive about 87 miles round trip to see one, and the rural heart-disease death rate is 21% higher than in cities. A nurse practitioner at a small clinic, with no stress-test lab, must decide which patients with chest pain to send on that trip.

## What we built
An offline browser tool (English, Spanish, Russian). **In:** 7 things a small clinic can measure: age, sex, chest-pain type, blood pressure, cholesterol, fasting glucose and resting ECG (the last three can be blank). **Out:** a green / yellow / red zone with an action, the factors behind it, and a step-by-step points table that shows exactly how the score was calculated. Real unseen patients can be loaded with their true angiography result, including one the model gets wrong. An interactive chart shows all 678 unseen test patients with a draggable threshold.

## What we added
Starting from suggested project 4.1.2, we (1) chose our own user, a rural clinic, and limited the model to the inputs it actually has; (2) asked our own question: does it work in hospitals it has never seen? We trained only on Cleveland and tested on Hungary, Switzerland and Long Beach VA; (3) audited fairness by sex and age and fixed the gap we found; (4) built a remote follow-up demo (synthetic patients) where a doctor sees a queue sorted by urgency.

## How it works
**Data:** UCI Heart Disease, 920 patients from 4 hospitals. Missing values and impossible zeros (e.g. cholesterol = 0) are treated as missing and filled with training medians. Target: >50% narrowing of a coronary artery on angiography. Cleveland split 80/20 (242 train, 61 test). **Model:** logistic regression on the 7 inputs, chosen because it can be explained per patient and runs offline. **Threshold:** chosen on training data only, as the highest cut-off that catches ≥90% of sick patients in 5-fold cross-validation (0.29).

## Results
Patients the model never saw. FN = sick patients missed, FP = false alarms.

| Method | Cleveland test (28 sick / 33 healthy) | 3 new hospitals (370 sick / 247 healthy) |
|---|---|---|
| Baseline: most common class ("nobody is sick") | 28 FN · 0 FP · accuracy 0.54 | 370 FN · 0 FP · accuracy 0.40 |
| Baseline: one rule ("refer if no chest pain") | 6 FN · 5 FP · accuracy 0.82 | 83 FN · 65 FP · accuracy 0.76 |
| **Ours: 7-input clinic model** | **2 FN · 9 FP · recall 0.93 · accuracy 0.82** | **43 FN · 87 FP · recall 0.88 · accuracy 0.79** |
| Textbook 13-input model | 0 FN · 11 FP · recall 1.00 | **93 FN** · 58 FP · recall 0.75 |

**Main takeaway:** the 13-input model looks perfect at home but misses more than twice as many sick patients in new hospitals. It needs stress-test and imaging results that are missing there, and filling them with Cleveland's typical values ("0 blocked vessels", "normal scan") makes every patient look healthy. This held in all 100 random re-splits we tried. The measurements that matter most are chest-pain type, sex and age.

**Fairness:** with one threshold the model caught 92% of sick men but only 53% of sick women. A separate threshold for women (0.07), set by a rule fixed on training data (every group must reach ≥90% recall in cross-validation), cut missed women from 15 to 5. The cost is 19 extra false alarms among women; men are unchanged. Caveat: only 32 sick women are in the test, and we chose this option after seeing all four we compared.

**Where it fails:** 35 of 398 sick patients still land in the green zone, mostly women or patients under 50 with atypical pain and a normal ECG. The model also copied a data artefact: in this 1980s data, typical angina looked *lower* risk than no pain. The app warns about both. Raw scores under-predict where disease is common (Switzerland: 0.64 average vs 93% sick), so we show zones, not percentages.

## What we'd do next
Validate on present-day US data with many more women, through a public dataset or an IRB-approved rural partner. Test the tool with practising rural clinicians.

## Sources
UCI Heart Disease (Janosi, Steinbrunn, Pfisterer & Detrano, 1988), https://doi.org/10.24432/C52P4X (CC BY 4.0). scikit-learn, pandas, NumPy, Matplotlib. CDC NCHS Data Brief 417 (2021); American College of Cardiology (2024), counties without a cardiologist. AI tool: Claude (Anthropic), https://claude.ai, which helped write code, text and translations; reviewed by the team. Full details, figures and the notebook are in the GitHub repository.
