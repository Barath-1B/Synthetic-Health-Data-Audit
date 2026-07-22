## Leakage Disclosure

During development, we discovered that PHQ9_TOTAL was included in the
feature set passed to downstream classifiers. Because Depression_Severity
is derived deterministically from PHQ9_TOTAL via fixed clinical thresholds,
this created a data-leakage scenario in which the classifier recovered an
arithmetic identity rather than modeling depression from independent features.

Pre-fix TRTR AUC (Random Forest, OVR macro): 0.9999
Post-fix TRTR AUC (Random Forest, OVR macro): 0.7125

All results reported in this paper use the corrected feature set.
Individual PHQ-9 item responses (DPQ010–DPQ090) are also excluded to
prevent indirect reconstruction of the target via summation. The NHANES
feature set used for generation and evaluation consists of demographic,
medical, dietary, alcohol, and sleep variables only (23 features total).

Feature set (post-fix, 23 features):
  Gender, Age, Race, Race_Ext, Education, Marital_Status, Income_Ratio,
  Arthritis, Congestive_Heart_Failure, Coronary_Heart_Disease, Heart_Attack,
  Stroke, Cancer, Sleep_Hours_Weekday, Sleep_Hours_Weekend,
  Trouble_Sleeping_Doc, Ever_Had_Drink, Drinking_Freq, Avg_Drinks_Per_Day,
  Binge_Drinking_Freq, Total_Calories, Total_Sugar, Total_Caffeine
