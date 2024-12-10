import pandas as pd
import numpy as np

data = pd.DataFrame({
    'Participant': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4', 'P5', 'P5', 'P6', 'P6'],

    'Task': ['Visual Task', 'Textual Task'] * 6,
    'RMSSD': [68.75, 74.60, 47.29, 40.52, 42.21, 31.71, 35.01, 35.19,76.662,67.742,66.579,52.765]

})

# Encoding tasks as numeric values
data['Task'] = data['Task'].map({'Visual Task': 0, 'Textual Task': 1})

# model
from statsmodels.formula.api import mixedlm

model = mixedlm("RMSSD ~ Task", groups=data["Participant"], data=data)
result = model.fit()


print(result.summary())


#         Mixed Linear Model Regression Results
# ======================================================
# Model:            MixedLM Dependent Variable: RMSSD
# No. Observations: 12      Method:             REML
# No. Groups:       6       Scale:              26.8359
# Min. group size:  2       Log-Likelihood:     -40.0643
# Max. group size:  2       Converged:          Yes
# Mean group size:  2.0
# ------------------------------------------------------
#            Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
# ------------------------------------------------------
# Intercept(participant)  56.084    7.045  7.960 0.000  42.275 69.892
# Task       -5.662    2.991 -1.893 0.058 -11.524  0.200
# Group Var 270.985   49.104
# ======================================================


