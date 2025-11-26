import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# log10スケールで表示
# ==================================================
def show_sar_log(img, title):
    img = np.squeeze(img)
    img_log = np.log10(np.abs(img) + 1e-10)
    vmin = np.percentile(img_log, 1)  # 下位1%
    vmax = np.percentile(img_log, 99)  # 上位99%
    plt.imshow(img_log, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')

# ==================================================
# Load files
# ==================================================
den_path = r"F:\wavelet\haar\real\denoised_Chiba1_band36_0_0.npy"
ref_path = r"F:\ver5_comp_umakuittayatu\results\real\denoised_Chiba1_band36_0_0.npy"

den = np.load(den_path)
ref = np.load(ref_path)

print("den :", den.shape)
print("ref :", ref.shape)

# ==================================================
# 表示（2枚比較）
# ==================================================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
show_sar_log(den, "noisy")

plt.subplot(1, 2, 2)
show_sar_log(ref, "Denoised")

plt.tight_layout()
plt.show()