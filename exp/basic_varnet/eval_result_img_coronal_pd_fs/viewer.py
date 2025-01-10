import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# PNG 파일 경로
image_path = "/home/intern1/pytorch_mri_variationalnetwork/exp/basic_varnet/eval_result_img_coronal_pd_fs/1.png"

# 이미지 로드 및 표시
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')  # 축 숨기기
plt.show()

