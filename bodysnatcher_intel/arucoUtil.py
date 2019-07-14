import unireedsolomon.rs as rs
import cv2.aruco as aruco
import numpy as np

ARUCO_HEIGHT = 5
ARUCO_WIDTH = 4


# Aruco corners start top left (0) and go clockwise (0..3).
#
# The desired ordering of arucos is by rows left-right and then top-down. As
#  usual, the largest checkerboard size (8 vs 5) is assumed 'horizontal'. Note
# that, confusingly, the larger checkerboard size for our config also has
# fewer arucos (4) vs 5 for the vertical...
#
# The row/col values correspond to a significant grid point that we use
# for comparisons. The internal grid is 4x7, i.e., the 28 points that we used
# to estimate the pose.
#
# The 'corner'  field chooses which corner of each aruco we use for the
# proximity test.
#
# The following table was created by visual inspection...
#
RECIPE_ORDERING = [
    { 'corner': 3, 'row': 0, 'col': 0}, # 0
    { 'corner': 3, 'row': 0, 'col': 2}, # 1
    { 'corner': 3, 'row': 0, 'col': 4}, # 2
    { 'corner': 3, 'row': 0, 'col': 6}, # 3

    { 'corner': 2, 'row': 1, 'col': 0}, # 4
    { 'corner': 2, 'row': 1, 'col': 2}, # 5
    { 'corner': 2, 'row': 1, 'col': 4}, # 6
    { 'corner': 2, 'row': 1, 'col': 6}, # 7

    { 'corner': 3, 'row': 2, 'col': 0}, # 8
    { 'corner': 3, 'row': 2, 'col': 2}, # 9
    { 'corner': 3, 'row': 2, 'col': 4}, # 10
    { 'corner': 3, 'row': 2, 'col': 6}, # 11

    { 'corner': 2, 'row': 3, 'col': 0}, # 12
    { 'corner': 2, 'row': 3, 'col': 2}, # 13
    { 'corner': 2, 'row': 3, 'col': 4}, # 14
    { 'corner': 2, 'row': 3, 'col': 6}, # 15

    { 'corner': 0, 'row': 3, 'col': 0}, # 16
    { 'corner': 0, 'row': 3, 'col': 2}, # 17
    { 'corner': 0, 'row': 3, 'col': 4}, # 18
    { 'corner': 0, 'row': 3, 'col': 6}  # 19
];

# type arucos is list of {id: int32, index: int32, corners: np.array(4,2)
#                                                           float32}
# type of gridPoints is np.array(N,2) float32
def fillSequence(arucos, gridPoints, nGridColumns):
    def findClosest(corner, point):
        aruco = None
        minVal = 1000000000000
        for x in arucos:
             d = np.linalg.norm(x['corners'][corner] - point)
             if d < minVal:
                 aruco = x
                 minVal = d
        return (minVal, aruco)

    def findOrdering() :
        return [findClosest(x['corner'], gridPoints[x['row']*nGridColumns +
                                                    x['col']])
                for x in RECIPE_ORDERING]

    def filterDuplicates(values):
        hash = {} # index-> (distance: number, position: number)
        for i, (d, aruco) in enumerate(values):
            if aruco['index'] in hash:
                prevD, _ = hash[aruco['index']]
                if (prevD > d):
                    hash[aruco['index']] = (d, i)
            else:
                hash[aruco['index']] = (d, i)
        return [aruco if hash[aruco['index']][1] == i else {'id' : -1}
                 for i, (_, aruco) in enumerate(values)]

    return filterDuplicates(findOrdering())

def splitSequence(all):
    erasure = []
    ids = []
    for i, x in enumerate(all):
        arucoId = x['id']
        if arucoId == -1:
            ids.append(0)
            erasure.append(i)
        else:
            ids.append(arucoId)
    return (ids, erasure)

def computeId(ids, erasure):
    def fromStr(a):
        return [ord(x) for x in a]

    def toStr(a):
        return ''.join([str(chr(x)) for x in a])

    # Compared to libfec:
    # (python) generator 3 maps to prim 13 in libfec
    # (python) fcr 1 maps to  fcr 19 in libfec
    # found after exhaustive search...
    #
    # The rest is similar: gfpoly=0x25, symbolsize=2^5, 20 symbols per block,
    # i.e., 12 parity and 8 data (encoding a 40bit id).


    coder = rs.RSCoder(20, 8, 3, 0x25, 1, 5)
    print 'Number of erasures: {0}'.format(len(erasure))
    try:
        corrected, ecc = coder.decode(toStr(ids), k=8, nostrip=True,
                                      erasures_pos=erasure)
        ecc = fromStr(ecc)
        corrected = fromStr(corrected)
        print corrected, ecc
        all = corrected + ecc
        if not coder.check(toStr(all)):
            print 'Cannot validate ecc'
            return None
        else:
            result = 0
            for i, x in  enumerate(corrected):
                result = result + x
                if i < len(corrected)-1:
                    result = result * 32 # 2^5
            return result
    except Exception as inst:
        print 'Cannot recover id: ', inst
        return None

# corners is (N,1,2) with shapeCorners (nRows, nCols), i.e., N=nRows*nCols
def findId(image, corners, shapeCorners):
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters_create()
    arucoCorners, arucoIds, _ = aruco.detectMarkers(image, arucoDict,
                                                    parameters=params)
    #arucoCorners is a list of np.arrays with shape (1, 4, 2) float32
    #arucoIds is an np.array of shape (N, 1) int32
    #print arucoCorners, arucoCorners[0].shape, arucoIds, arucoIds.dtype

    if arucoIds is None:
        return None

    allArucos = [{'id': x[0], 'index': i, 'corners': arucoCorners[i][0]}
                 for i, x in enumerate(arucoIds)]

    if len(allArucos) > 0:
        seq = fillSequence(allArucos,
                           np.reshape(corners, (corners.shape[0], 2)),
                           shapeCorners[1])

        #print seq
        ids, erasure = splitSequence(seq)
        print ids, erasure

        return computeId(ids, erasure)
    else:
        return None
