import numpy as np
import cv2
import itertools
import math
import sys
import base64

import arucoUtil as aruco

import pyrealsense2 as rs

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

#FULL_FRAME_WIDTH = 1920
#FULL_FRAME_HEIGHT = 1080
FULL_FRAME_WIDTH = FRAME_WIDTH
FULL_FRAME_HEIGHT = FRAME_HEIGHT

CHESS_HEIGHT = 4
CHESS_WIDTH = 7

ALL_CHESS = CHESS_HEIGHT*CHESS_WIDTH

MAX_ERROR = 0.01 # 1cm

def initCamera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT,
                         rs.format.z16, 30)
    config.enable_stream(rs.stream.color, FULL_FRAME_WIDTH, FULL_FRAME_HEIGHT,
                         rs.format.bgr8, 30)
    profile = pipeline.start(config)
    deviceDepth = profile.get_device().first_depth_sensor()
    deviceDepth.set_option(rs.option.visual_preset, 4) # high density
    print deviceDepth.get_option_value_description(rs.option.visual_preset, 4)
    depthScale = deviceDepth.get_depth_scale()

    # align color to depth
    align = rs.align(rs.stream.depth)

    return (pipeline, depthScale, align)


def warmUp(pipeline):
    count = 0
    while count < 50:
        frames = pipeline.wait_for_frames()
        count = count + 1

# left right first, then top to bottom
# Coordinates origin in top center, looking out, i.e., X axis positive
# to the left, Y axis positive down, Z axis positive to front, i.e., using
# right-hand coordinates.
#
# The units are in meters.
#
# The grid is 7x4, i.e.,  CHESS_WIDTHxCHESS_HEIGHT
refPoints3D = np.array([
    [ 0.1855,  0.082 ,  0.    ],
    [ 0.133 ,  0.082 ,  0.    ],
    [ 0.0805,  0.082 ,  0.    ],
    [ 0.028 ,  0.082 ,  0.    ],
    [-0.0245,  0.082 ,  0.    ],
    [-0.077 ,  0.082 ,  0.    ],
    [-0.1295,  0.082 ,  0.    ],
    [ 0.1855,  0.1345,  0.    ],
    [ 0.133 ,  0.1345,  0.    ],
    [ 0.0805,  0.1345,  0.    ],
    [ 0.028 ,  0.1345,  0.    ],
    [-0.0245,  0.1345,  0.    ],
    [-0.077 ,  0.1345,  0.    ],
    [-0.1295,  0.1345,  0.    ],
    [ 0.1855,  0.187 ,  0.    ],
    [ 0.133 ,  0.187 ,  0.    ],
    [ 0.0805,  0.187 ,  0.    ],
    [ 0.028 ,  0.187 ,  0.    ],
    [-0.0245,  0.187 ,  0.    ],
    [-0.077 ,  0.187 ,  0.    ],
    [-0.1295,  0.187 ,  0.    ],
    [ 0.1855,  0.2395,  0.    ],
    [ 0.133 ,  0.2395,  0.    ],
    [ 0.0805,  0.2395,  0.    ],
    [ 0.028 ,  0.2395,  0.    ],
    [-0.0245,  0.2395,  0.    ],
    [-0.077 ,  0.2395,  0.    ],
    [-0.1295,  0.2395,  0.    ]
], dtype="double")

def toMatrix(rotMat, trans):
    transV = trans.reshape(3, 1)
    out = np.hstack((rotMat, transV))
    out = np.vstack((out, np.array([[0, 0, 0, 1.0]])))
    outInv = np.linalg.inv(out)
    mat = np.transpose(out).reshape(16).tolist()
    matInv =  np.transpose(outInv).reshape(16).tolist()
    return mat, matInv


def compute3DPoints(corners, depth, depthScale, intrinsics):
    result = []
    for c in corners:
        xy = c[0]
        x = int(round(xy[0]))
        y = int(round(xy[1]))
        d = depth[y, x] * depthScale #meters
        if d != 0:
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            result.append(p)
        else:
            return False, np.array(result)

    return True, np.array(result)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# corners shape is (N,1,2)
def orderCorners(corners):
    nCorners = corners.shape[0]
    distance = np.linalg.norm(corners[1:] -corners[:-1], axis=2)
    shortScan = np.sum(distance[CHESS_HEIGHT-1::CHESS_HEIGHT])/(1.0*
                                                                (CHESS_WIDTH-1))
    longScan = np.sum(distance[CHESS_WIDTH-1::CHESS_WIDTH])/(1.0*
                                                             (CHESS_HEIGHT-1))
    if shortScan > longScan:
        print 'Changing scan order'
        print shortScan, longScan

        # Short side was the principal side to scan because
        # it found the south-west corner first, and started scanning bottom to
        # top, using the skinny side.
        #
        # Otherwise, it would have scanned top to bottom (left-to right) using
        # the wide side.
        #
        # Need to convert...
        top2Bot = np.flip(np.reshape(corners, [CHESS_WIDTH, CHESS_HEIGHT, 2],
                                     order='c'), axis =1)
        return np.transpose(top2Bot, (1,0,2)).reshape(nCorners, 1, 2)
    else:
        return corners

def mainSnapshot(options=None):
    pipeline, depthScale, align = initCamera()
    try:
        warmUp(pipeline)
        while True:
            frames = pipeline.wait_for_frames()
            alignedFrames = align.process(frames)
            depth = alignedFrames.get_depth_frame()
            color = alignedFrames.get_color_frame()
            if depth and color:
                depth_intrin = depth.profile.as_video_stream_profile().intrinsics
                print depth_intrin
                colorImage = np.asanyarray(color.get_data())
                    # no flip       regArray = np.flip(regArray, 1)
#                imgRGB = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)
                ret, buf = cv2.imencode(".jpg", colorImage)
                if ret == True:
                    data = base64.b64encode(buf)
                    return {'width': FRAME_WIDTH, 'height': FRAME_HEIGHT,
                            'data': data}
                else:
                    print 'fail'
    finally:
        pipeline.stop()

def toFlatList(p):
    return list(itertools.chain(*p.tolist()))

# inputs shape is (N,3)
# find rotation+translation that maps refPoints3D to points3D with minimal error
def arun(refPoints3D, points3D):
    centroidRef = np.mean(refPoints3D, axis=0)
    refPOrigin = refPoints3D - centroidRef
    centroid = np.mean(points3D, axis=0)
    pOrigin = points3D - centroid
    #covariance matrix
    cov = np.dot(np.transpose(refPOrigin), pOrigin)
    #print "covariance shape {0}".format(cov.shape)
    u, s, vh = np.linalg.svd(cov)
    # 'vh' is already transposed, need to undo it
    rot = np.dot(vh.T, u.T)
    if np.linalg.det(rot) < 0:
        print "reflection corrected"
        vh[2, :] = -vh[2, :]
        rot =  np.dot(vh.T, u.T)
    trans = -np.dot(rot, centroidRef) + centroid
    err = np.linalg.norm(np.dot(rot, refPoints3D.T).T + trans - points3D,
                         axis=1)
    print err
    err = np.mean(err)
    print err
    return err, rot, trans


def mainCalibrate(options=None):
    pipeline, depthScale, align = initCamera()
    try:
        warmUp(pipeline)
        while True:
            print '.'
            frames = pipeline.wait_for_frames()
            alignedFrames = align.process(frames)
            depth = alignedFrames.get_depth_frame()
            color = alignedFrames.get_color_frame()
            if depth and color:
                intrin = depth.profile.as_video_stream_profile().intrinsics
                colorImage = np.asanyarray(color.get_data())
                gray = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (CHESS_HEIGHT,
                                                                CHESS_WIDTH),
                                                         None)
                if ret == True:
                    print 'got it'
                    print corners
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11),
                                               (-1,-1), criteria)
                    #corners2 shape is (N,1,2)
                    corners2 = orderCorners(corners2)
                    depthImage = np.asanyarray(depth.get_data())
                    ok, points3D = compute3DPoints(corners2, depthImage,
                                                   depthScale, intrin)
                    #points3D shape is (N,3)
                    if ok == True:
                        print corners2
                        print points3D
                        if (options and options['onlyPoints3D'] == True):
                            result = {'points3D': toFlatList(points3D),
                                      'shape3D': [CHESS_HEIGHT, CHESS_WIDTH]}
                            return result
                        else:
                            err, rot, trans = arun(refPoints3D, points3D)
                            if err < MAX_ERROR:
                                uid = aruco.findId(gray, corners2,
                                                   (CHESS_HEIGHT, CHESS_WIDTH))
                                if uid != None:
                                    mat, invMat = toMatrix(rot, trans)
                                    print 'Rotation {0}'.format(rot)
                                    print 'Translation {0}'.format(trans)
                                    result = {#'rotation': toFlatList(rot),
                                              #'translation': trans.tolist(),
                                              'uid': uid,
                                              # From external to our coordinates
                                              'transformMat': mat,
                                              # From our coordinates to external
                                              'inverseTransformMat': invMat,
                                              'shape3D': [CHESS_HEIGHT,
                                                          CHESS_WIDTH],
                                              'points3D': toFlatList(points3D)
                                    }
                                    return result
                                else:
                                    print 'Cannot find UID'
                            else:
                                print 'Max reprojection error exceeded'
                    else:
                        print 'Missing depth for corner'
    finally:
        pipeline.stop()

if __name__ == "__main__":
    mainCalibrate()
