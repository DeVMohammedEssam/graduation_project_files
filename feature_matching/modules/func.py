import numpy as np
import cv2 as cv


def find_good_matches(img1, img2):
    '''
    @return -> (good matches list , keypoinst of img1, keypoints of img2 )
    @brief this function compares between two images and return the good mathced features between them using SIFT detector and flann based matcher

    @param img1 query Image i.e. the image to be compared
    @param img2 train Image i.e. the Image to be compared to.
    '''
    # initiate sift detector
    sift = cv.xfeatures2d.SIFT_create()

    # get keypoints and descriptors
    kp1, dsc1 = sift.detectAndCompute(img1, None)
    kp2, dsc2 = sift.detectAndCompute(img2, None)

    # creating falnn matcher
    FLANN_INDEX_KDTREE = 0
    index_params = {
        'algorithm': FLANN_INDEX_KDTREE,
        'trees': 5
    }
    search_params = {'checks': 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # get matches
    matches = flann.knnMatch(dsc1, dsc2, k=2)

    # filter only good matches using ration test
    # if the distance of the 1st match is less than 2nd matche's distance then it is a good match
    good = []
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            good.append(m1)
    return (good, kp1, kp2)

    """ 
    ###############################################################################################
    """


def draw_on_match(good, img1, kp1, img2, kp2, minMatchCount=10):
    '''
     @return -> the image after drawing

     @brief Draws a polyline around the mathced image (the one that has the most mathces number)

     @param good list of the good matches
     @param minMatchCount the minimum matches number to start drawing  is equal to 10 by default
     @param img1 query Image i.e. the image to be compared
     @param img2 train Image i.e. the Image to be compared to.
     @param kp1 keypoints of img1
     @param kp2 keypoints of img2

    '''
    if len(good) > minMatchCount:
        # getting source image position points
        # .reshape(-1,1,2) => to convert the src points to numpy 3d array
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        retval, mask = cv.findHomography(
            src_pts, dst_pts, method=cv.RANSAC, confidence=0.5)
        # getting the mask of the matches
        matchesMask = mask.ravel().tolist()
        # getting height and width of the queryImage
        h, w = img1.shape
        # setting drawing points and reshape it to 3d array
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        # get the drawing precpective
        dst = cv.perspectiveTransform(pts, retval)
        img2 = cv.polylines(img2, [np.int32(dst)],
                            True, 255, 3, lineType=cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=(None),
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    output = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return output
