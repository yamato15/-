import cv2
import matplotlib.pyplot as plt
import urllib.request as req

url = "https://jp.static.photo-ac.com/assets/img/ai_page/ai_model_512_01.png"
req.urlretrieve(url, "face_test.png")

img = cv2.imread("face_test.png")
print(img.shape)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()