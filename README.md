# Information Criterion Selection
Code for model selection via five information criteria: Akaike's information criterion (AIC), the corrected AIC for finite length 
observations (AICc), Bayesian information criterion (BIC), two-stage Minimum Description Length (MDL) and normalized Minimum Description 
Length (nMDL). The dictionaries are the Frequency (Fourier) dictionary and the Frequency + Spike dictionary. Model selection is performed
with Orthogonal Matching Pursuit or the simple Threshold method.

For the nMDL criterion, we used the approximation from the paper "Model Selection and the Principle of Minimum Description Length", Mark 
Hansen and Bin Yu, Journal of the American Statistical Association, Volume 96, Issue 454, 2001. The OMP-based algorithms can be improved,
for instance, by performing optimisation over the sine and cosine bases instead of the (complex-valued) Discrete Fourier Transform 
dictionary. Currently, it is possible that the OMP algorithms return models that are complex-valued.
