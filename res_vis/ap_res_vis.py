"""

	F-Voxnet	Swin-T wo InterTrans	Swin-T with InterTrans	R-50 wo InterTrans	R-50 with InterTrans
FPS	8.333333333	9.515372154	9.515372154	14.31029699	14.31029699
AP50	37.275	36.6125	52.70625	29.775	42.975
#param (M)	101	85	85	82	82
FLOPs (G)	819	745	745	739	739
"""
import matplotlib.pyplot as plt

plt.box(False)

methods = [
    "F-Voxnet",
    "Swin-T wo InterTrans",
    "Swin-T with InterTrans",
    "R-50 wo InterTrans",
    "R-50 with InterTrans",
]
FPS = [8.333333333, 9.515372154, 9.515372154, 14.31029699, 14.31029699]
AP50 = [37.275, 36.6125, 52.70625, 29.775, 42.975]
params = [101, 85, 85, 82, 82]
# For better vis
params_ = [101 * 1.5, 85 * 1, 85 * 1, 82 * 0.95, 82 * 0.95]
# params_round = [round(p/10) for p in params]
params_round = [p * 5 for p in params_]
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
