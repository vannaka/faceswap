"""
The script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

import sys

import cv2
import numpy
import dlib


#------------------------------------------------------------------------------
#   GLOBAL CONSTANTS
#------------------------------------------------------------------------------
PREDICTOR_PATH = "training_data/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS
                + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during color correction, as a fraction of the
# pupillary distance.
COLOR_CORRECT_BLUR_FRAC = 0.6

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)




#------------------------------------------------------------------------------
#   Main Method
#------------------------------------------------------------------------------
def main():
    """
    Main function.
    
    image_one is the image we are overlaying the face onto.
    image_two is the image we are taking the face from.

    """

    # Video source
    VIDEO = cv2.VideoCapture(0)

    # Read in the images
    image_one = get_image(sys.argv[1])
    image_one = cv2.resize(image_one, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

    while True:
        ret, frame = VIDEO.read()

        # Reduce image size by 25% to reduce processing time and improve framerates
        frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

        # flip image so that it's more mirror like
        frame = cv2.flip(frame, 1)

        # Swap the faces
        output_im = swap_faces(frame, image_one)

        # Show the swapped faces
        cv2.imshow('FaceSwaper TM', output_im)

        # cv2.imshow('Camera feed', frame)
        
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    # Cleanup
    VIDEO.release()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------
#   EXCEPTIONS
#------------------------------------------------------------------------------
class NoFaces(Exception):
    """Exception raised if there are no faces in a given image."""
    pass


#------------------------------------------------------------------------------
#   Swap faces
#------------------------------------------------------------------------------
def swap_faces(image_one, image_two):
    """Swap one set of faces in the input images"""

    # Get the locations of facial features
    try:
        landmarks_one = get_landmarks(image_one)
    except NoFaces:
        print("No faces detected in image one")
        image_one = cv2.resize(image_one, None, fx=1.33, fy=1.33, interpolation=cv2.INTER_LINEAR)
        return image_one

    try:
        landmarks_two = get_landmarks(image_two)
    except NoFaces:
        print("No faces detected in image two")
        image_one = cv2.resize(image_one, None, fx=1.33, fy=1.33, interpolation=cv2.INTER_LINEAR)
        return image_one

    # Get the translation matrix to align the face in image two to the face in image one
    trans_matrix = get_trans_matrix(landmarks_one, landmarks_two)
    # trans_matrix = transformation_from_points(landmarks_one[ALIGN_POINTS],
    #                                           landmarks_two[ALIGN_POINTS])

    # Get a mask of the faces in both images
    mask_one = get_face_mask(image_one, landmarks_one)
    mask_two = get_face_mask(image_two, landmarks_two)

    # translate mask to position of face in image one
    warped_mask_two = warp_image(mask_two, trans_matrix, image_one.shape)

    # Combine the warped mask two with mask one to get a mask that covers
    #   both faces when they are overlaid.
    combined_mask = numpy.max([mask_one, warped_mask_two], axis=0)

    # Translate image two such that its face aligns with the face in image one
    warped_image_two = warp_image(image_two, trans_matrix, image_one.shape)

    # Correct color of seconod image so the facial colors match
    warped_corrected_im2 = correct_colors(image_one, warped_image_two, landmarks_one)

    # Combine the face from image two with image one
    output_im = (image_one * (1.0 - combined_mask)) + (warped_corrected_im2 * combined_mask)

    # Convert to proper format
    output_im = (254 * (output_im - output_im.min())) / (output_im.max() - output_im.min())
    output_im = output_im.astype(numpy.uint8)

    # Upscale since the passed in images were downscaled to 75%
    output_im = cv2.resize(output_im, None, fx=1.33, fy=1.33, interpolation=cv2.INTER_LINEAR)

    return output_im


#------------------------------------------------------------------------------
#   get_landmarks
#------------------------------------------------------------------------------
def get_landmarks(image):
    """Get coordinates of facial features."""
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    rects = DETECTOR(image, 1)

    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in PREDICTOR(image, rects[0]).parts()])


#------------------------------------------------------------------------------
#   annotate_landmarks
#------------------------------------------------------------------------------
def annotate_landmarks(im, landmarks):
    """Produce an image with the ladmarks overlaid on it."""

    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


#------------------------------------------------------------------------------
#   read_im_and_landmarks
#------------------------------------------------------------------------------
def get_image(fname):
    """Read in an image."""

    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image.shape[1] * SCALE_FACTOR,
                         image.shape[0] * SCALE_FACTOR))

    return image


#------------------------------------------------------------------------------
#   draw_convex_hull
#------------------------------------------------------------------------------
def draw_convex_hull(im, points, color):
    """"""
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


#------------------------------------------------------------------------------
#   get_face_mask
#------------------------------------------------------------------------------
def get_face_mask(im, landmarks):
    """
    Take image and landmarks and creates a mask of the eyes, nose, brows and mouth.
    """
    # Create image of all zeros.
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    # Draw two convex polygons in white.
    #   One around the eye area and one around the nose and mouth area.
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    # Convert image to rgb from grayscale.
    #   R, G, and B are all set to the same value. This produces the same color
    #   as in the grayscale image.
    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    # Blur the image to hide any discontinuities between the two convex hulls generated above.
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def get_trans_matrix(points1, points2):
    points1 = points1.astype(numpy.float32)
    points2 = points2.astype(numpy.float32)

    src = numpy.array([numpy.mean(points1[LEFT_EYE_POINTS], axis=0),
           numpy.mean(points1[RIGHT_EYE_POINTS], axis=0),
           numpy.mean(points1[MOUTH_POINTS], axis=0)])

    dst = numpy.array([numpy.mean(points2[LEFT_EYE_POINTS], axis=0),
           numpy.mean(points2[RIGHT_EYE_POINTS], axis=0),
           numpy.mean(points2[MOUTH_POINTS], axis=0)])

    return cv2.getAffineTransform(src, dst)

#------------------------------------------------------------------------------
#   warp_im
#------------------------------------------------------------------------------
def warp_image(image, trans_matrix, dshape):
    """
    Translate image using the given translation matrix.
    """

    # Create image of all zeros.
    output_image = numpy.zeros(dshape, dtype=image.dtype)

    cv2.warpAffine(image,
                   trans_matrix[:2],
                   (dshape[1], dshape[0]),
                   dst=output_image,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_image


#------------------------------------------------------------------------------
#   correct_colors
#------------------------------------------------------------------------------
def correct_colors(im1, im2, landmarks1):
    """"""
    blur_amount = COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))

    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


# Call the main method
main()
