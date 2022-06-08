# hand embedding
import mediapipe as mp
import numpy as np

class Embedder(object):
    def __init__(self):
        self._landmark_names = mp.solutions.hands.HandLandmark
        # <HandLandmark.WRIST: 0>,
        # <HandLandmark.THUMB_CMC: 1>,
        # <HandLandmark.THUMB_MCP: 2>,
        # <HandLandmark.THUMB_IP: 3>,
        # <HandLandmark.THUMB_TIP: 4>,
        # <HandLandmark.INDEX_FINGER_MCP: 5>,
        # <HandLandmark.INDEX_FINGER_PIP: 6>,
        # <HandLandmark.INDEX_FINGER_DIP: 7>,
        # <HandLandmark.INDEX_FINGER_TIP: 8>,
        # <HandLandmark.MIDDLE_FINGER_MCP: 9>,
        # <HandLandmark.MIDDLE_FINGER_PIP: 10>,
        # <HandLandmark.MIDDLE_FINGER_DIP: 11>,
        # <HandLandmark.MIDDLE_FINGER_TIP: 12>,
        # <HandLandmark.RING_FINGER_MCP: 13>,
        # <HandLandmark.RING_FINGER_PIP: 14>,
        # <HandLandmark.RING_FINGER_DIP: 15>,
        # <HandLandmark.RING_FINGER_TIP: 16>,
        # <HandLandmark.PINKY_MCP: 17>,
        # <HandLandmark.PINKY_PIP: 18>,
        # <HandLandmark.PINKY_DIP: 19>,
        # <HandLandmark.PINKY_TIP: 20>
    def __call__(self, landmarks):
        # modify the call func can both handle a 3-dim dataset and a single referencing result.
        if  isinstance(landmarks, np.ndarray):
            if landmarks.ndim == 3: # for dataset
                embeddings = []
                for lmks in landmarks:
                    embedding = self.__call__(lmks)
                    embeddings.append(embedding)
                return np.array(embeddings)
            elif landmarks.ndim == 2: # for inference
                assert landmarks.shape[0] == len(list(self._landmark_names)), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])
                # Normalize landmarks.
                landmarks = self._normalize_landmarks(landmarks)
                # Get embedding.
                embedding = self._get_embedding(landmarks)
                return embedding
            else:
                print('ERROR: Can NOT embedding the data you provided !')
        else:
            if isinstance(landmarks, list): # for dataset
                embeddings = []
                for lmks in landmarks:
                    embedding = self.__call__(lmks)
                    embeddings.append(embedding)
                return np.array(embeddings)
            elif isinstance(landmarks, mp.framework.formats.landmark_pb2.NormalizedLandmarkList): # for inference
                # Normalize landmarks.
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark], dtype=np.float32)
                assert landmarks.shape[0] == len(list(self._landmark_names)), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])
                landmarks = self._normalize_landmarks(landmarks)
                # Get embedding.
                embedding = self._get_embedding(landmarks)
                return embedding
            else:
                print('ERROR: Can NOT embedding the data you provided !')

    def _get_center(self, landmarks):
        # MIDDLE_FINGER_MCP:9
        return landmarks[9] 

    def _get_size(self, landmarks):
        landmarks = landmarks[:, :2]
        max_dist = np.max(np.linalg.norm(landmarks - self._get_center(landmarks), axis=1))
        return max_dist * 2

    def _normalize_landmarks(self, landmarks):
        landmarks = np.copy(landmarks)
        # Normalize
        center = self._get_center(landmarks)
        size = self._get_size(landmarks)
        landmarks = (landmarks - center) / size
        landmarks *= 100  # optional, but makes debugging easier.
        return landmarks

    def _get_embedding(self, landmarks):
        # we can add and delete any embedding features
                # we can add and delete any embedding features
        test = np.array([
            np.dot((landmarks[2]-landmarks[0]),(landmarks[3]-landmarks[4])),   # thumb bent
            np.dot((landmarks[5]-landmarks[0]),(landmarks[6]-landmarks[7])),
            np.dot((landmarks[9]-landmarks[0]),(landmarks[10]-landmarks[11])),
            np.dot((landmarks[13]-landmarks[0]),(landmarks[14]-landmarks[15])),
            np.dot((landmarks[17]-landmarks[0]),(landmarks[18]-landmarks[19]))
        ]).flatten()      
#         print(test)
        return test