import cv2
import numpy as np
import matplotlib.pyplot as plt
# newImageInfo = (500,500,3)
#
# dst = np.zeros(newImageInfo,np.uint8)
# cv2.rectangle(dst,(50,100),(200,300),(255,0,0),5)
# cv2.circle(dst,(250,250),(50),(0,255,0),2)
# cv2.ellipse(dst,(256,256),(150,100),0,0,180,(255,255,0),-1)
# points = np.array([[150,50],[140,140],[200,170],[250,250],[150,50]],np.int32)
# points = points.reshape(-1,1,2)
# cv2.polylines(dst,[points],True,(0,255,255))
# cv2.imshow("dst",dst)
# cv2.waitKey(0)

# a = [1735.486, 847.4552, 1662.6808, 843.02783, 1672.371, 683.67786,
#  1745.1761, 688.1052 ]
# # print(len(a))
# xmin = min(a[0],a[2],a[4],a[6])
# xmax = max(a[0],a[2],a[4],a[6])
# ymin = min(a[1],a[3],a[5],a[7])
# ymax = max(a[1],a[3],a[5],a[7])
# plt.figure(figsize=(24,13.5),dpi=80)
# ax = plt.gca()
# ax.invert_yaxis()
# plt.scatter(a[0],a[1],color='red')
# plt.scatter(a[2],a[3],color='blue')
# plt.scatter(a[4],a[5],color='yellow')
# plt.scatter(a[6],a[7],color='green')
# plt.scatter(xmin,ymin,color='purple',marker='*')
# plt.scatter(xmax,ymax,color='orange',marker='*')
# plt.show()

a = 3.552
print(int(round(a)))