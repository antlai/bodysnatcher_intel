#import logging
#logging.basicConfig(level=logging.DEBUG)
import cProfile
import time
import numpy as np
import cv2
import sys
import json
import pyrealsense2 as rs
import constants

import bodyparts as bp

def scale(inMat):
    a = 255.0 /(constants.MAX_DISTANCE-constants.MIN_DISTANCE)
    b = -a * constants.MIN_DISTANCE
    scaled = np.clip(inMat, constants.MIN_DISTANCE,
                     constants.MAX_DISTANCE) * a + b
    return scaled.astype(np.uint8)

def unscale(inMat):
    a = (constants.MAX_DISTANCE-500.0)/255.0
    b = constants.MIN_DISTANCE
    unscaled = inMat * a + b
    unscaled = unscaled.astype(np.float32)
    unscaled[unscaled == constants.MAX_DISTANCE] = constants.MAX_FLOAT32
    return unscaled

def paintContour(contours, hierarchy, minVal):
    maxArea = 0.;
    c = -1
    for idx, con in enumerate(contours):
        area = cv2.contourArea(con)
        if area > constants.MIN_CONTOUR_AREA:
            c = idx if area > maxArea else c
            maxArea = area if area > maxArea else maxArea

    #  Draw only the biggest and its holes
    alpha = np.zeros(minVal.shape, dtype=np.uint8)
    if c >= 0:
        cv2.drawContours(alpha, [contours[c]], 0, 255, -1)
        for index, child in  enumerate(hierarchy[0]):
            if child[3] == c:
                if cv2.contourArea(contours[index]) > constants.MIN_HOLE_AREA:
                    #print 'hole', index, c
                    cv2.drawContours(alpha, [contours[index]], 0, 0, -1)
    return alpha

#background subtraction
def subtraction(options, minVal, d):
    options = None
    if (options and options.get('display')):
        print ('#unknowns ', str(d[d == 0.].size))
    if (options and options.get('display')):
                cv2.imshow('Before subtraction', d / 4500.)
    d[d == 0.] = constants.MAX_FLOAT32
    #cv2.imshow('Before subtraction', d / 4500.)
    #         cv2.waitKey(1)
    #foregroundSize = d.size - d[d>=(minVal-50.)].size
    d[d>=(minVal-constants.DISTANCE_THRESHOLD)] = constants.MAX_FLOAT32
    #print count, foregroundSize
    #cv2.imshow('After subtraction', d / 4500.)
    if (options and options.get('display')):
        cv2.imshow('After subtraction', d / 4500.)
    ret, thr = cv2.threshold(scale(d), 200, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('Threshold', thr)
    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)
    return paintContour(contours, hierarchy, minVal)

def initCamera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, constants.FRAME_WIDTH,
                         constants.FRAME_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, constants.FULL_FRAME_WIDTH,
                         constants.FULL_FRAME_HEIGHT, rs.format.bgr8, 30)
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

def inpaint(options, minVal):
    #inpainting the background
    mask = np.zeros(minVal.shape, dtype=np.uint8)
    mask[minVal == constants.MAX_FLOAT32] = 255
    if (options and options.get('display')):
        cv2.imshow('before', minVal / 4500.)

#    cv2.imshow('before', minVal / 4500.)
#    cv2.waitKey(0)
    if (options and options.get('display')):
        cv2.imshow('mask', mask)

    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    scaledMin = scale(minVal)
    scaledPatchedMin = cv2.inpaint(scaledMin,mask,3,cv2.INPAINT_TELEA)
    approxMinVal = unscale(scaledPatchedMin)
    #print (approxMinVal[mask == 255])
    minVal[mask == 255] = approxMinVal[mask == 255]

    if (options and options.get('display')):
        cv2.imshow('after', minVal / 4500.)

#    cv2.imshow('after', minVal / 4500.)
#    cv2.waitKey(0)
    return minVal

def computeBackground(options, pipeline, align):
    count = 0
    minVal = np.full((constants.FRAME_DECIMATED_HEIGHT,
                      constants.FRAME_DECIMATED_WIDTH),
                     constants.MAX_FLOAT32, dtype=np.float32)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 4)

    temporal = rs.temporal_filter()
    while count < 60:
        frames = pipeline.wait_for_frames()
        alignedFrames = align.process(frames)
        depth = alignedFrames.get_depth_frame()
#        depth = frames.get_depth_frame()

        filtered_depth = decimation.process(depth)
#        filtered_depth = temporal.process(filtered_depth)
#        filtered_depth = depth

        d = np.asanyarray(filtered_depth.get_data()).astype(np.float32)
        print d.shape, d.dtype
        zeros = d.size - np.count_nonzero(d)
        print('Input:zeros:' + str(zeros) + ' total:' + str(d.size))
        d[d == 0.] = constants.MAX_FLOAT32
        minVal = np.minimum(minVal, d)
        print ('Minval: zeros:' +
               str(minVal[minVal == constants.MAX_FLOAT32].size) +
               ' total:' + str(minVal.size))
        count = count + 1

    return inpaint(options, minVal)

def mainSegment(options = None):
    pipeline, depthScale, align = initCamera()
    print '--------------------------------------------'
    print options
    counter = 0
    if constants.PROFILE_ON:
        pr  = cProfile.Profile() #profile
        pr.enable() #profile
    try:
        warmUp(pipeline)
        minVal = computeBackground(options, pipeline, align)
        net = bp.newNet()
        t0 =  time.time()
        counterOld = counter
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude,
                              constants.DECIMATION_FACTOR)
        temporal = rs.temporal_filter(0.5, 20, 5)#valid 1 of last 2
        while True:
            counter = counter + 1
            frames = pipeline.wait_for_frames()
#            alignedFrames = align.process(frames)
#            depth = alignedFrames.get_depth_frame()
            depth = frames.get_depth_frame()
            filtered_depth = decimation.process(depth)
            filtered_depth = temporal.process(filtered_depth)
            # intrinsics already reflects decimation
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics

#            filtered_depth = depth
#            if counter % 2 == 1:
                #help the temporal filter since we cannot process full rate
#                continue
            d = np.asanyarray(filtered_depth.get_data()).astype(np.float32)
            alpha = subtraction(options, minVal, d)
            if (options and options.get('display')):
                cv2.imshow('Contour', alpha)
            if constants.PROFILE_ON:
                result = pr.runcall(bp.process, options, net, d, alpha,
                                    depthScale, intrinsics) #profile
            else:
                result = bp.process(options, net, d, alpha, depthScale,
                                    intrinsics)
            if (options and options.get('display')):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if counter % 10 == 0:
                t1 = time.time()
                print '{:.3f} images/sec'.format((counter - counterOld)/(t1-t0))
                t0 = t1
                counterOld = counter
            yield json.dumps(result)
    finally:
        if constants.PROFILE_ON:
            pr.disable()
            pr.print_stats()#profile
        pipeline.stop()


def loop(options = None):
    g = mainSegment(options)
    for res in g:
        print (res)

if __name__ == "__main__":
    mainSegment()
#    cProfile.run('mainSegment()') #mainSegment()
