import numpy as np

N =  23326  #Number of testing instances
z = 1.96#error/accuracy
W_F1 = .913#F1 score or accuracy
print((W_F1*(1-W_F1))/N)
print(z**2/(4*(N**2)))
diff_term = z * (np.sqrt((W_F1*(1-W_F1))/N + z**2/(4*(N**2))))
lower_bound = float(N)/(N + z**2) * (W_F1 + float(z**2)/(2*N) - diff_term)
upper_bound = float(N)/(N + z**2) * (W_F1 + float(z**2)/(2*N) + diff_term)
fix = float(N)/(N + z**2) * (W_F1 + float(z**2)/(2*N))
var = float(N)/(N + z**2) * diff_term
print("high bound:" + str(fix) + str(' +/- ') + str(var))
