# -------------------------------------- profile_detection ---------------------------------------
detect_frontal_face = 'profile_detection/haarcascades/haarcascade_frontalface_alt.xml'
detect_perfil_face = 'profile_detection/haarcascades/haarcascade_profileface.xml'

# -------------------------------------- emotion_detection ---------------------------------------
# emiton's model
path_model = 'emotion_detection/Modelos/model_dropout.hdf5'
# model's parameters
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


#define the aspect ratio of the EAT eye
# number of consecutive frames
EYE_AR_THRESH = 0.23 #baseline
EYE_AR_CONSEC_FRAMES = 1

# eye landmarks
eye_landmarks = "blink_detection/model_landmarks/shape_predictor_68_face_landmarks.dat"