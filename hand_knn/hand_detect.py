import cv2
import mediapipe as mp
import numpy as np
from hand_knn.embedder import Embedder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def init_knn(file='D:\Academy\CIVIL-459 DLAV\Project\deep_sort_pytorch/hand_knn/dataset_embedded.npz'):
    # Load and proprocessing dataset
    # TODO 修改路径！！
    npzfile = np.load(file)
    X = npzfile['X']
    y = npzfile['y']
    # KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X,y)
    # Embedder
    embedder = Embedder()
    return neigh, embedder

def hand_pose_recognition(stream_img, neigh, embedder):
  # For static images: 
  stream_img = cv2.cvtColor(stream_img, cv2.COLOR_BGR2RGB)   
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:

          
          results = hands.process(stream_img)
          if not results.multi_hand_landmarks:
            return 'no_gesture', stream_img
          else:
            h, w, _ = stream_img.shape
            annotated_image = stream_img.copy()
            multi_landmarks = results.multi_hand_landmarks
            # KNN inference
            embeddings = embedder(multi_landmarks)
            hand_class = neigh.predict(embeddings) # digit class
            hand_class_prob = neigh.predict_proba(embeddings)
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image,
                                            landmarks,
                                            mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())
            
            return hand_class, annotated_image