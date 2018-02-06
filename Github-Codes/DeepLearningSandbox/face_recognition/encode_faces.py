"""Encode faces into a descriptor

  The model used here was trained such that if the Euclidian
  distance between two face decriptors is less than 0.6
  then they are from the same person, otherwise, they are from
  different people.

  This 0.6 threshold achieves a 99.3% accuracy on the Labelled
  Faces in the Wild face recognition challenge.
"""
from __future__ import print_function

import sys
import os
import glob
import argparse

from skimage import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

import dlib


D_PLOT = 1       #flag to debug by viewing feature space in a 2D space
D_DIM = 128      #number of dims for the descriptor from compute_face_descriptor()
FILE_EXT = "jpg" #require all images to end in jpg
NB_COMPONENTS=2  #PCA components


def encode(detector, shape_predictor, model, image, win=None):
  """Encodes faces from a single image into a 128 dim descriptor.

  Args:
    detector: dlib face detector object
    shape_predictor: dlib shape predictor object
    model: dlib convnet model
    image: image as numpy array
    win: dlib window object for vizualization if VIZ flag == 1

  Returns:
    list of descriptors (np array) for each face detected in image
  """
  # dlib comments:
  # Ask the detector to find the bounding boxes of each face. The 1 in the
  # second argument indicates that we should upsample the image 1 time. This
  # will make everything bigger and allow us to detect more faces.
  dets = detector(img, 1)
  print("Number of faces detected: {}".format(len(dets)))

  descriptors = []
  for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = sp(img, d)
    # Draw the face landmarks on the screen so we can see what face is currently being processed.

    if win is not None:
      win.clear_overlay()
      win.set_image(img)
      win.add_overlay(d)
      win.add_overlay(shape)
      dlib.hit_enter_to_continue()

    # Compute the 128D vector that describes the face in img identified by shape
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    descriptors.append(np.asarray(list(face_descriptor)))

  return descriptors


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--landmarks_dat", default="shape_predictor_68_face_landmarks.dat",
      help="download at http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
  a.add_argument("--model", default="dlib_face_recognition_resnet_model_v1.dat",
      help="download at http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
  a.add_argument("--image", help="images must be .jpg files")
  a.add_argument("--image_dir", help="images must be .jpg files")
  a.add_argument("--viz_off", action='store_true', help="flag to vizualize detections and shape prediction")

  args = a.parse_args()

  if args.landmarks_dat is None or args.model is None or (args.image is None and args.image_dir is None):
    a.print_help()
    sys.exit(1)

  detector = dlib.get_frontal_face_detector()
  sp = dlib.shape_predictor(args.landmarks_dat)
  facerec = dlib.face_recognition_model_v1(args.model)

  win = None
  if not args.viz_off:
    win = dlib.image_window()

  if args.image is not None:
    img = io.imread(args.image)
    print(encode(detector, sp, facerec, img, win))

  if args.image_dir is not None:
    fns = os.listdir(args.image_dir)
    descriptors = []
    for fn in fns:
      if FILE_EXT not in fn:
        continue

      img = io.imread(os.path.join(args.image_dir, fn))
      descriptors.append(encode(detector, sp, facerec, img, win))

    if D_PLOT == 1:
      pca = PCA(n_components=NB_COMPONENTS)
      X = np.squeeze(np.asarray(descriptors))
      X_r = pca.fit(X).transform(X)
      plt.figure()
      plt.scatter(X_r[:,0], X_r[:,1])
      plt.show()

