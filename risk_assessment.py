import numpy as np
import cv2
import matplotlib.pyplot as plt 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D


def measure(seg, var, x, y, z):
    '''
    Measurement function for each image
    Input: 
     One-Hot Encoded Segmentation CxWxH
     Classwise Uncertainty Map CxWxH
     Corresponding X,Y,Z - Maps 
    Returns: 
    Dict of measurements
    {
        corrosion:
            Instance 1: Area, Area Var, Depth Var
            .
            .
            .
            Instance n: Area, Area Var, Depth Var
        crack:
            Instance 1: Length, Length Var, Depth Var
            .
            .
            .
            Instance n: Length, Length Var, Depth Var
        
    }
    '''
    def __area__(cnt):
        '''
        Helper function to measure defect area
        '''
        defect = np.zeros_like(predPitting)
        cv2.drawContours(defect, [cnt], 0, 255, 1)
        pixelIndices = np.where(defect == 255)
        xList = x[pixelIndices]
        yList = y[pixelIndices]
        zList = z[pixelIndices]
        pointsList = np.transpose(np.asarray([xList, yList, zList]))
        pointsList = pointsList[~np.all(pointsList == 0, axis = 1)]
        hull = ConvexHull(pointsList)
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        area = hull.area
        return area

    def __depth__(cnt):
        '''
        Helper function to measure defect depth
        Required args: contour instance
        Returns: average depth
        '''
        defect = np.zeros_like(predPitting)
        cv2.drawContours(defect, [cnt], 0, 255, 1)
        pixelIndices = np.where(defect == 255)
        xList = x[pixelIndices]
        yList = y[pixelIndices]
        zList = z[pixelIndices]
        # Compute average depth - VERIFY
        print("Depth Calculation -- VERIFICATION NEEDED --")
        depth = np.mean(z[pixelIndices])
        return depth

    def __length__(cnt):
        '''
        Helper function to measure defect length
        '''
        x,y,w,h = cv2.boundingRect(cnt) # TODO: Which is the var for length here? 
        # TODO: Convert pixel coordinates to camera coordinates
        return length


    measurements = {'corrosion': {}, 'crack': {}}
    predPitting = seg[1]
    predCrack = seg[2]
    # TODO: Thresholding step for defects (Morph ops?)
    
    # Pitting Defects
    contours, hierarchy = cv2.findContours(predPitting, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key = lambda x: cv2.arcLength(x, True))
    for i, cnt in enumerate(contoursSorted):
        area = __area__(cnt)
        depth = __depth__(cnt)
        measurements['corrosion'][i] = [area, depth] # TODO: uncertainty
    
    # Cracking defects
    contours, hierarchy = cv2.findContours(predCrack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key = lambda x: cv2.arcLength(x, True))
    for i, cnt in enumerate(contoursSorted):
        # Length of the crack is approximated using the bounding rectangle of the contour
        length = __length__(cnt)
        depth = __depth__(cnt)
        measurements['crack'][i] = [length, depth]
    return measurements
