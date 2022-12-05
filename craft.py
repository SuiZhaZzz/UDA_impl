import numpy as np
import cv2

mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
to_rgb = True

# mean = np.float32(mean.reshape(3, 1, 1))
mean = np.broadcast_to(mean, (2,3,2,2))
print(mean.dtype)

img = np.random.randint(10, size=(2,3,2,2)).astype(np.float32)
print(img.dtype)
print(img)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
# print(img.shape)

# # for i in range(2):
# #     t = img[i]
# #     print(t)
# #     t = t.transpose(1,2,0).copy().astype(np.float32)
# #     cv2.subtract(t, mean, t)
# #     t = t.transpose(2,0,1)
# #     print(t)

# img = img.transpose(0,2,3,1).copy().astype(np.float32)
# print(img.shape)
# print(mean.shape)

cv2.subtract(img, mean, img)

# img = img.transpose(0,3,1,2)
# print(img)

# cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
print(img)