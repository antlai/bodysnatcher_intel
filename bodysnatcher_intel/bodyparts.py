import cv2
import numpy as np
import sys
#import caffe
import math
import os
import keras_model as ke
import constants

# Models from "Real-Time Human Motion Capture with Multiple Depth Camera"
#   by Alireza Shafaei and  James J. Little
#     UBC, Vancouver, Canada
# https://github.com/ashafaei/dense-depth-body-parts.git

def scale(inMat):
    a = 255.0 /(constants.MAX_DISTANCE-constants.MIN_DISTANCE)
    b = -a * constants.MIN_DISTANCE
    return np.clip(inMat, constants.MIN_DISTANCE,
                   constants.MAX_DISTANCE) * a + b

def compressVector():
    res = np.arange(constants.COLOR_MAP.shape[0], dtype=np.uint8)
    for index, val in constants.COMPRESS_FRONT:
        res[index] = val
    return res

COMPRESS_VECTOR = compressVector()


#returns [left,right) and [top, bottom) (not including `right` and `bottom`)
def boundaries(alpha):
    psum_width = alpha.sum(axis=0)
    ind = psum_width.nonzero()
    if ind[0].size == 0:
        return (False, ())
    else:
        left = ind[0][0]
        right = ind[0][-1]
        psum_height = alpha.sum(axis=1)
        if ind[0].size == 0:
            return (False, ())
        else:
            ind = psum_height.nonzero()
            top = ind[0][0]
            bottom = ind[0][-1]
            return (True, (left, right+1, top, bottom+1))

#respect aspect ratio
def newDimensions(height, width):
    #print height, width
    if height > width:
        newHeight = constants.INNER_WINDOW
        newWidth = int(round(width * (newHeight/ float(height))))
    else:
        newWidth = constants.INNER_WINDOW
        newHeight =  int(round(height * (newWidth/ float(width))))
    return (newHeight, newWidth)

def crop(data, alpha, box):
    left, right, top, bottom = box
    width = right-left;
    height = bottom-top;
    newHeight, newWidth = newDimensions(height, width)
    data = data[top:bottom, left:right]
    alpha = alpha[top:bottom, left:right]
    return (data, alpha, newWidth, newHeight)

def resize(data, alpha, newWidth, newHeight):
    newData = cv2.resize(data, (newWidth, newHeight),
                         interpolation = cv2.INTER_LANCZOS4)
    newAlpha = cv2.resize(alpha, (newWidth, newHeight),
                          interpolation = cv2.INTER_NEAREST)
    return (newData, newAlpha)


# output is float [0, 1.0] with background=1.0
def normalizeData(data, alpha):
    meanDepth = data[alpha == 255].mean()
    # note that 55 is 1.6m + 0.5m offset in a range [0.5, 8.0]...
    data[alpha == 255] = np.clip(data[alpha == 255] +
                                 (55.0 - meanDepth), 0,
                                 255).astype(np.float32)
    data[alpha != 255] = 255.
    data = data /255.0
    return data

def place(data, alpha, newWidth, newHeight):
    finalData = np.ones((constants.WINDOW_SIZE, constants.WINDOW_SIZE),
                        dtype=np.float32)
    finalAlpha = np.zeros((constants.WINDOW_SIZE, constants.WINDOW_SIZE),
                          dtype=np.uint8)
    locH = int(np.floor((constants.WINDOW_SIZE - newHeight)/2.))+1
    locW = int(np.floor((constants.WINDOW_SIZE - newWidth)/2.))+1
    finalData[locH:locH+newHeight, locW:locW+newWidth] = data
    finalAlpha[locH:locH+newHeight, locW:locW+newWidth] = alpha
    return finalData, finalAlpha, locW, locH

#use tiling to stride inside the cache
def blockArgmax(mat):
    # input is [46,INNER_WINDOW , INNER_WINDOW] np.float
    #output is [WINDOW_SIZE, WINDOW_SIZE] np.uint8
    res = np.ones((constants.WINDOW_SIZE, constants.WINDOW_SIZE),
                  dtype = np.uint8)*constants.BACKGROUND_INDEX
    for rB in range(constants.NUM_BLOCKS):
        for rC in range(constants.NUM_BLOCKS):
            blMat = mat[:,rB*constants.SIZE_BLOCK:(rB+1)*constants.SIZE_BLOCK,
                        rC*constants.SIZE_BLOCK:(rC+1)*constants.SIZE_BLOCK]
            s = np.argmax(np.copy(np.transpose(blMat, (1,2,0)),order='C'),
                          axis=2).astype(np.uint8)
            s = COMPRESS_VECTOR[s]

            res[constants.MARGIN+rB*constants.SIZE_BLOCK:
                constants.MARGIN+(rB+1)*constants.SIZE_BLOCK,
                constants.MARGIN+rC*constants.SIZE_BLOCK:
                constants.MARGIN+(rC+1)*constants.SIZE_BLOCK] = s
    return res

# resolve left/right confusion by ignoring the minority ones in a bimodal dist
def filterLeftRight(mat):
    c = mat[:,1]
    maxC = np.max(c)
    minC = np.min(c)
    a = 255.0/(maxC-minC)
    b = -a*minC
    newC = (a*c+b+0.5).astype(dtype = np.uint8)
    thr, thMat = cv2.threshold(newC, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nonzeros = np.count_nonzero(thMat)
    zeros = thMat.size - nonzeros
    mapThr = int(0.5 + (thr - b)/a)
    mask = mat[:,1] > mapThr
    return mat[mask] if (nonzeros > zeros) else  mat[np.logical_not(mask)]

def deproject_pixel_to_point(intrinsics, pixel, depth):
    c, r = pixel
    x = (c - intrinsics.ppx) / float(intrinsics.fx)
    y = (r - intrinsics.ppy) / float(intrinsics.fy)
    depth = float(depth)
    return [depth*x, depth*y, depth]

def centroids_approx(depth, depthScale, intrinsics, all):
    #median point in 2-D, then project.
    #similar when partitions at roughly the same distance and convex
    m = np.median(all, 0)
    m = m.astype(np.int32)
#    print m
    r, c = m[0], m[1]
    d = depth[r, c] *depthScale #meters
    if  (d > constants.MIN_DISTANCE_METERS) and \
        (d < constants.MAX_DISTANCE_METERS):
        p = deproject_pixel_to_point(intrinsics, [c, r], d)
        return (p[0], p[1], p[2])
    else:
        #this is a rare event, inner areas are usually ok
        # second chance
        r = r -1 if r > 0 else r
        c = c -1 if c > 0 else c
        d = depth[r, c] *depthScale #meters
        if  (d > constants.MIN_DISTANCE_METERS) and \
            (d < constants.MAX_DISTANCE_METERS):
            p = deproject_pixel_to_point(intrinsics, [c, r], d)
            return (p[0], p[1], p[2])
        else:
            print 'No depth for centroid'
            return None

def centroids_slow(depth, depthScale, intrinsics, all):
    points3D_X = 0.0
    points3D_Y = 0.0
    points3D_Z = 0.0
    count = 0
    allList = all.tolist()
    for r, c in allList:
        d = depth[r, c] *depthScale #meters
        if  (d > constants.MIN_DISTANCE_METERS) and \
            (d < constants.MAX_DISTANCE_METERS):

            p = deproject_pixel_to_point(intrinsics, [c, r], d)
#            print 'r={0:d} c={1:d} d={2:f} x={3:f} y={4:f} z={5:f}'.format(r, c,d,p[0],p[1],p[2])

            points3D_X = points3D_X + p[0]
            points3D_Y = points3D_Y + p[1]
            points3D_Z = points3D_Z + p[2]
            count = count + 1
    if count == 0:
        return None
    else:
        return (points3D_X/count, points3D_Y/count, points3D_Z/count)

# def centroids(registration, undistorted, all):
#     points3D = registration.getPointArrayXYZ(undistorted, all)
#     points3D = points3D[~np.isnan(points3D[:, 0])]
#     mean = points3D.mean(0).tolist();
#     return (mean[0], mean[1], mean[2])

def analyze(mat, box,  depth, depthScale, intrinsics):
    # input is [FRAME_HEIGHT, FRAME_WIDTH] np.uint8
    #output is [(part, (mean_X, mean_Y, mean_Z))]
    left, right, top, bottom = box
    target =  mat[top:bottom, left:right]
    values, counts = np.unique(target, return_counts=True)
    pairs = zip(values, counts)
    pairs = [(x,y) for (x, y) in pairs if y>constants.MIN_COUNT and
             x != constants.BACKGROUND_INDEX]
    result = []

    for part, _ in pairs:
        #inefficient in general, but typically just a few body parts, e.g., <10
        all = np.argwhere(target == part) + np.array([top, left])
        all = np.int32(all)
        if (part in constants.BIMODAL_SET):
            all = filterLeftRight(all)
        centre = centroids_approx(depth, depthScale, intrinsics, all)
        if centre != None:
            result.append((int(part), centre))

    return result


def inference(options, net, data, alpha):
#    if (options and options.get('display')):
#        cv2.imshow('smallInput', data)
    data =  np.reshape( np.float32(data), [1, 250, 250,1])
    classes = net.predict(data)
#    print classes.shape
    #classes =  np.argmax(classes, axis=-1)#delete

    classes = np.reshape(classes, (250, 250))
    classes = COMPRESS_VECTOR[classes]
    classes[alpha == 0] = constants.BACKGROUND_INDEX
    return classes

#    net.blobs['data'].data[...] = data
#    output = net.forward()
#   output_prob = np.copy(output['prob'][0]) #slow to reference caffe blobs
    #    denseSmall = np.argmax(output_prob, axis=0)
#    dense = blockArgmax(output_prob)
#    dense = np.zeros(((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
#    print output['prob'].shape,  output['prob'][0]
#    dense =  output['prob'][0][0]
#    dense[alpha == 0] = constants.BACKGROUND_INDEX
#    return dense

def newNet():
    net =  ke.loadNetwork()
    data = np.zeros([1, constants.WINDOW_SIZE, constants.WINDOW_SIZE, 1],
                    dtype=np.float32)
    net.predict(data) #lazy loading of model
    return net

def toColorImage(result, projectedInfo):
    keyPoints = [cv2.KeyPoint(p[0], p[1], constants.KP_SIZE) for (x,p) in
                 projectedInfo]
    newImage = constants.COLOR_MAP[result].astype(np.uint8)  # / 255.
    # RGB to BGR
    img =  newImage[:,:,::-1]
    return cv2.drawKeypoints(img, keyPoints, None, color = (0, 0, 0))

def undoPlace(result, placeLoc, afterScaleDim):
    locW, locH = placeLoc
    width, height = afterScaleDim
    return result[locH:locH+height, locW:locW+width]

def undoResize(result, box):
    left, right, top, bottom = box
    width = right-left;
    height = bottom-top;
    return cv2.resize(result, (width, height),
                      interpolation = cv2.INTER_NEAREST)

def undoCrop(result, box, originalDim):
    left, right, top, bottom = box
    width, height = originalDim
    big = np.ones((height, width), dtype=np.uint8) * constants.BACKGROUND_INDEX
    big[top:bottom, left:right] = result
    return big

def undoTransforms(result, placeLoc, afterScaleDim, box, originalDim):
    result = undoPlace(result, placeLoc, afterScaleDim)
    result = undoResize(result, box)
    return undoCrop(result, box, originalDim)

def project(info, intrinsics):
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0,             0, 1]], dtype = np.float)
    def projectOne(p):
        x, y, z = p
        u, v, l =  np.matmul(camera_matrix, np.array([[x],[y],[z]])).tolist()
        u = int(round(u[0] / l[0]))
        v = int(round(v[0] / l[0]))
        return (u,v)

    return map(lambda (t,p): (int(t), projectOne(p)), info)

def process(options, net, depth, alpha, depthScale, intrinsics):
    height, width = depth.shape
    scaledDepth = scale(depth)
    #print width, height
    #print alpha.shape
    ok, box = boundaries(alpha)
    #print ok, box
    if ok == True:
        scaledDepth, alpha, newWidth, newHeight = crop(scaledDepth, alpha, box)
        scaledDepth, alpha = resize(scaledDepth, alpha, newWidth, newHeight)
        scaledDepth = normalizeData(scaledDepth, alpha)
        scaledDepth, alpha, locW, locH = place(scaledDepth, alpha, newWidth,
                                               newHeight)
        result = inference(options, net, scaledDepth, alpha)

        big = undoTransforms(result, (locW, locH), (newWidth, newHeight), box,
                             (width, height))
        #print(intrinsics)
        info = analyze(big, box, depth, depthScale, intrinsics)
        #print (info)
        projectedInfo = project(info, intrinsics)
        if (options and options.get('display')):
            cv2.imshow('bigOutput', toColorImage(big, projectedInfo))

        return (info, projectedInfo)
    else:
        return None
