"""

	F-Voxnet	Swin-T wo simCrossTrans	Swin-T with simCrossTrans	R-50 wo simCrossTrans	R-50 with simCrossTrans
FPS    9.091		9.515372154	9.515372154	14.31029699	14.31029699
AP50	37.275	36.6125	52.70625	29.775	42.975
#param (M)	64.1	48	48	45	45
"""
import matplotlib.pyplot as plt

plt.box(False)

methods = [
    "F-Voxnet",
    "Swin-T w/o simCrossTrans",
    "Swin-T with simCrossTrans",
    "R-50 w/o simCrossTrans",
    "R-50 with simCrossTrans",
]
FPS = [9.091, 9.515372154, 9.515372154, 14.31029699, 14.31029699]
AP50 = [37.275, 36.6125, 52.70625, 29.775, 42.975]
params_ = [64.1, 48, 48, 45.0, 45.0]
# For better vis
# params_round = [round(p/10) for p in params]
params_round = [p * 8 for p in params_]
print(params_round)
conv_color = "#66bd63"
trans_color = "#f46d43"
colors = [conv_color, trans_color, trans_color, conv_color, conv_color]
# plt.scatter(FPS, AP50, s = 80*5, c = colors, alpha=0.8)
plt.scatter(FPS, AP50, s=params_round, c=colors, alpha=0.98)
plt.xlabel("FPS")
plt.ylabel("mAP@IoU=.5")
plt.title("2D detection based on SUN RGB-D dataset")
plt.show()
