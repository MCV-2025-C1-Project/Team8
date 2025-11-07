"""
you haKeypoint descriptor implementations.
https://amroamroamro.github.io/mexopencv/matlab/cv.ORB.detectAndCompute.html
"""
import cv2 as cv


def orb_descriptor(img, nfeatures=500):
    """
    Compute ORB keypoints and descriptors.
    
    Returns:
        keypoints The detected keypoints. A 1-by-N structure array with the following fields:
            pt coordinates of the keypoint [x,y]
            size diameter of the meaningful keypoint neighborhood
            angle computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees and measured relative to image coordinate system (y-axis is directed downward), i.e in clockwise.
            response the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling.
            octave octave (pyramid layer) from which the keypoint has been extracted.
            class_id object class (if the keypoints need to be clustered by an object they belong to).
        descriptors Computed descriptors. Output concatenated vectors of descriptors. Each descriptor is a 32-element vector, as returned by cv.ORB.descriptorSize, so the total size of descriptors will be numel(keypoints) * obj.descriptorSize(), i.e a matrix of size N-by-32 of class uint8, one row per keypoint.
    """
    orb = cv.ORB.create(
        nfeatures=nfeatures,
    )
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def sift_descriptor(img, nfeatures=500):
    """
    Compute SIFT keypoints and descriptors.

    SIFT produces float32 descriptors that capture gradient distributions around
    each keypoint, which typically match best with L2 distance.
    """
    if hasattr(cv, "SIFT_create"):
        sift = cv.SIFT_create(
            nfeatures=nfeatures
        )
    else:
        # Fall back to xfeatures2d module if the build does not expose SIFT at the top-level
        sift = cv.xfeatures2d.SIFT_create(
            nfeatures=nfeatures
        )
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors
