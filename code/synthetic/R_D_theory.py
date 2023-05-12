from dit.rate_distortion import RDCurve, IBCurve
import dit
import matplotlib.pyplot as plt

d = dit.Distribution(['00', '02', '12', '21', '22'], [1/5]*5)
fig = plt.figure()
IBCurve(d, beta_num=26).plot()
import pdb; pdb.set_trace()