import math
import skimage
import torchaudio

from PRNet.run_basics import prn
from RetinaFace_Pytorch import eval_widerface
from RetinaFace_Pytorch import torchvision_model

from skimage.io import imread
from skimage.transform import rescale

from ABAW2020TNT.utils import ex_from_one_hot, split_EX_VA_AU
from ABAW2020TNT.tsav import TwoStreamAuralVisualModel
from ABAW2020TNT.clip_transforms import ComposeWithInvert, Normalize, AmpToDB, NumpyToTensor

from PIL import Image
import os
from cv2 import VideoCapture, imwrite, cvtColor, COLOR_BGR2RGB, circle, line, getRotationMatrix2D, LINE_8, polylines, \
    resize, INTER_CUBIC, warpAffine, IMWRITE_JPEG_QUALITY, COLOR_BGR2GRAY
from torch import load, device, from_numpy, cuda, hann_window
from torch import zeros as torch_zeros
from numpy import uint8, array, int32, degrees, arctan, rint, genfromtxt, zeros, clip, round


# Record a video during around 5 seconds
def record_video():
    cap = VideoCapture(0)
    count = 0
    return_layers = {'layer2': 1, 'layer3': 2, 'layer4': 3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = load('RetinaFace_Pytorch/model/model.pt', map_location=device('cpu'))

    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}

    RetinaFace.load_state_dict(pretrained_dict)
    RetinaFace.eval()
    while count < 8:
        _, frame = cap.read()
        imwrite(os.path.join("data/frames/input_frames/", str(count) + ".jpg"), frame)
        while not detect_face_one_image("data/frames/input_frames/" + str(count) + ".jpg", RetinaFace):
            _, frame = cap.read()
            imwrite(os.path.join("data/frames/input_frames/", str(count) + ".jpg"), frame)
            detect_face_one_image("data/frames/input_frames/" + str(count) + ".jpg", RetinaFace)
        count = count + 1
    cap.release()


def detect_face_one_image(img_name, RetinaFace):
    img = skimage.io.imread(img_name)
    img = from_numpy(img)
    img = img.permute(2, 0, 1)
    input_img = img.unsqueeze(0).float()
    picked_boxes, picked_landmarks, _ = eval_widerface.get_detections(input_img, RetinaFace)
    try:
        box = list(map(lambda x: int(x), picked_boxes[0][0]))

        h = box[3] - box[1]
        w = box[2] - box[0]
        img = img.cpu().permute(1, 2, 0).numpy()
        img = img[box[1]:box[1] + h, box[0]:box[0] + w]
        if not img.shape[0] < 112:
            img = cvtColor(img, COLOR_BGR2RGB)
            imwrite(img_name, img)
        return True
    except TypeError:
        return False


def predict(image):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > 1000:
        image = rescale(image, 1000. / max_size)
        image = (image * 255).astype(uint8)
    pos = prn.process(image)
    kpt = prn.get_landmarks(pos)
    return pos, kpt


def draw_kpts(img, kpt):
    end_list = array([17, 22, 27, 42, 48, 31, 36, 68], dtype=int32) - 1
    temp_image = img.copy()

    kpt = round(kpt).astype(int32)
    for i in range(kpt.shape[0]):

        st = kpt[i, :2]

        temp_image = circle(temp_image, (st[0], st[1]), 1, (0, 0, 255), 2)

        if i in end_list:
            continue
        ed = kpt[i + 1, :2]

        temp_image = line(temp_image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return temp_image


def _find_center_pt(points):
    '''
    find centroid point by several points that given
    '''
    x = 0
    y = 0
    num = len(points)
    for pt in points:
        x += pt[0]
        y += pt[1]
    x //= num
    y //= num
    return x, y


def _angle_between_2_pt(p1, p2):
    '''
    to calculate the angle rad by two points
    '''
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return degrees(arctan(tan_angle))


def _get_rotation_matrix(left_eye_pt, right_eye_pt, left_outer_mouth_points, right_outer_mouth_points, nose_center,
                         face_img, scale):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    mouth_angle = _angle_between_2_pt(left_outer_mouth_points, right_outer_mouth_points)
    angle = (eye_angle + mouth_angle) / 2
    M = getRotationMatrix2D((nose_center[0] / 2, nose_center[1] / 2), angle, scale)

    return M


def _get_rotation_matrix1(left_eye_pt, right_eye_pt, left_outer_mouth_points, right_outer_mouth_points, nose_center,
                          face_img, scale):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    M = getRotationMatrix2D((nose_center[0] / 2, nose_center[1] / 2), eye_angle, scale)
    return M


def draw_mask(points, imp):
    image = imp.copy()
    line_type = LINE_8
    left_eyebrow = points[17:22, :2]
    right_eyebrow = points[22:27, :2]
    nose_bridge = points[28:31, :2]
    chin = points[6:11, :2]
    mouth_outer = points[48:60, :2]
    left_eye = points[36:42, :2]
    right_eye = points[42:48, :2]
    pts = [rint(mouth_outer).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, True, color=(255, 255, 255), thickness=1, lineType=line_type)
    pts = [rint(left_eyebrow).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, False, color=(223, 223, 223), thickness=1, lineType=line_type)
    pts = [rint(right_eyebrow).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, False, color=(191, 191, 191), thickness=1, lineType=line_type)
    pts = [rint(left_eye).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, True, color=(159, 159, 159), thickness=1, lineType=line_type)
    pts = [rint(right_eye).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, True, color=(127, 127, 127), thickness=1, lineType=line_type)
    pts = [rint(nose_bridge).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, False, color=(63, 63, 63), thickness=1, lineType=line_type)
    pts = [rint(chin).reshape(-1, 1, 2).astype(int32)]
    polylines(image, pts, False, color=(31, 31, 31), thickness=1, lineType=line_type)
    return image


def analyse_images(images):
    aligned_faces_dir = "data/frames/aligned_faces"
    mask_dir = "data/frames/masks"
    cropped_face_output_dir = "data/frames/input_frames/"
    for image_name in images:
        img = imread(cropped_face_output_dir + image_name)
        img = resize(img, (112, 112), interpolation=INTER_CUBIC)
        pos, kpt = predict(img)
        left_eye_points = kpt[36:42, :2]
        right_eye_points = kpt[42:48, :2]
        nose_tip = kpt[30, :2]
        left_outer_mouth_points = kpt[48, :2]
        right_outer_mouth_points = kpt[54, :2]
        left_eye_center = _find_center_pt(left_eye_points)
        right_eye_center = _find_center_pt(right_eye_points)
        trotate = _get_rotation_matrix(left_eye_center, right_eye_center, left_outer_mouth_points,
                                       right_outer_mouth_points, nose_tip, img, scale=0.9)
        warped = warpAffine(img, trotate, (112, 112), flags=INTER_CUBIC, borderValue=0.0)
        warped = cvtColor(warped, COLOR_BGR2RGB)

        pos, kpt = predict(warped)

        imwrite(os.path.join(aligned_faces_dir, image_name), warped, [int(IMWRITE_JPEG_QUALITY), 95])
        black_frame = zeros((112, 112, 3), uint8)
        mask = draw_mask(kpt, black_frame)
        mask = cvtColor(mask, COLOR_BGR2GRAY)
        imwrite(os.path.join(mask_dir, image_name), mask, [int(IMWRITE_JPEG_QUALITY), 100])


def ex_to_str(arr):
    str = "{:d}".format(arr)
    return str


def select_gpu_cpu():
    if cuda.is_available():
        import backends.cudnn as cudnn
        cudnn.enabled = True
        dev = device("cuda")
    else:
        dev = device("cpu")
    return dev


def select_frame_video():
    # Pick frames of the video
    clip_len = 8
    width, height = 112, 112
    label_frame = 1
    dilation = 6
    _range = range(10 - label_frame + dilation, 10 - label_frame + dilation * (clip_len + 1), dilation)

    clip_transform = ComposeWithInvert([NumpyToTensor(), Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
                                                                   std=[0.22803, 0.22145, 0.216989, 0.225])])
    clip = zeros((clip_len, width, height, 4))

    image_list = os.listdir('data/frames/aligned_faces')
    for i in range(clip_len):
        img = Image.open(os.path.join('data/frames/aligned_faces', image_list[i]))
        mask = Image.open(os.path.join('data/frames/masks', image_list[i]))
        clip[i, :, :, 0:3] = array(img)
        clip[i, :, :, 3] = array(mask)

    video_data = clip_transform(clip)
    video_data = video_data.unsqueeze(0)
    return video_data


def audio():
    num_frames_video = 276
    with open('data/other/video_ts.txt', 'r') as f:
        time_stamps = genfromtxt(f)[:num_frames_video]
    sample_len_secs = 10
    sample_rate = 44100
    sample_len_frames = sample_len_secs * sample_rate
    window_size = 20e-3
    window_stride = 10e-3
    window_fn = hann_window
    audio_shift_sec = 5
    audio_shift_samples = audio_shift_sec * sample_rate
    num_fft = 2 ** math.ceil(math.log2(window_size * sample_rate))

    audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64,
                                                           n_fft=num_fft,
                                                           win_length=int(window_size * sample_rate),
                                                           hop_length=int(window_stride * sample_rate),
                                                           window_fn=window_fn)

    audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

    audio_file = 'data/audio/audio.wav'
    audio, sample_rate = torchaudio.load(audio_file, num_frames=min(sample_len_frames,
                                                                    max(int((time_stamps[100] / 1000) * sample_rate),
                                                                        int(window_size * sample_rate))),
                                         offset=max(int((time_stamps[
                                                             100] / 1000) * sample_rate - sample_len_frames + audio_shift_samples),
                                                    0))

    audio_features = audio_transform(audio).detach()
    if audio.shape[1] < sample_len_frames:
        _audio_features = torch_zeros((audio_features.shape[0], audio_features.shape[1],
                                       int((sample_len_secs / window_stride) + 1)))
        _audio_features[:, :, -audio_features.shape[2]:] = audio_features
        audio_features = _audio_features
        audio_features = audio_spec_transform(audio_features)
    audio_features = audio_features.unsqueeze(0)
    if audio.shape[1] < sample_len_frames:
        _audio = torch_zeros((1, sample_len_frames))
        _audio[:, -audio.shape[1]:] = audio
        audio = _audio
    return audio, audio_features


def return_data():
    audio_data, audio_features = audio()
    return {'clip': select_frame_video(), 'audio_features': audio_features, 'audio': audio_data}


def extract_emotion(data):
    model_path = 'ABAW2020TNT/model2/TSAV_Sub4_544k.pth.tar'
    model = TwoStreamAuralVisualModel(num_channels=4)
    # load the model
    saved_model = load(model_path, map_location=device('cpu'))
    model.load_state_dict(saved_model['state_dict'])
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)
    result = model(data)
    result = result.cpu()

    EX, VA, AU = split_EX_VA_AU(result)

    label_array = ex_from_one_hot(EX.numpy())
    label_array = clip(label_array, 0.0, 6.0)
    label_array = round(label_array).astype(int)
    writer = ex_to_str
    ans = writer(label_array[0])
    label_dict = {"0": "Neutral",
                  "1": "Anger",
                  "2": "Disgust",
                  "3": "Fear",
                  "4": "Happiness",
                  "5": "Sadness",
                  "6": "Surprise"}
    print(label_dict[ans])
    return label_dict[ans]


def video_emotion_recognition():
    record_video()
    analyse_images(os.listdir('data/frames/input_frames/'))
    extract_emotion(return_data())


if __name__ == "__main__":
    video_emotion_recognition()


