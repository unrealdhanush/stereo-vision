import numpy as np
import cv2
from matplotlib import pyplot as plt
import logging
from random import randrange as rr
logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10
MAX_FEATURES = 500

img1 = cv2.imread('example_1/input/image1.jpeg', 0)  # queryImage
img2 = cv2.imread('example_1/input/image2.jpeg', 0)  # trainImage
#img2 = img1

sift=cv2.SIFT_create(MAX_FEATURES)
# Compute SIFT keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
pts1=[]
pts2=[]
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    #Fundamental Matrix estimation
    F, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, F)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        color=(rr(255),rr(255),rr(255))
        cv2.line(newimg, pt1, pt2, color)
    plt.title("SIFT Keypoint Matches")
    plt.imshow(newimg)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask=None

draw_parameters=dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = matchesMask, flags=2)
newimg_inliersonly=cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_parameters)
plt.imshow(newimg_inliersonly)
plt.title("Inliers Only")
plt.show()

#grayimg1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#grayimg2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=160, blockSize=15)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.title("Disparity Map")
plt.show()