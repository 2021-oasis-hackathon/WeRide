import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
from skimage.transform import resize
from PIL import Image

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = Image.fromarray(image).resize(size=(160,80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks)) # 차선 mask 이미지

    # Re-size to match the original image
    # lane_image = np.array(Image.fromarray((lane_drawn * 255).astype(np.uint8)).resize((image.shape[1],image.shape[0])).convert('RGB'))
    # Merge the lane drawing onto the original image
    lane_image = resize(lane_drawn, (856, 480)).astype(np.uint8)
    img_resized = cv2.resize(lane_image, (image.shape[1], image.shape[0]))

    temp = [0, int(img_resized.shape[1] / 3), int(img_resized.shape[1] * 2 / 3), img_resized[0].shape]
    img_resized1 = np.zeros_like(img_resized[0:img_resized.shape[0], 0:temp[1]]).astype(np.uint8)
    roi = img_resized[0:img_resized.shape[0], temp[1]:temp[2]]
    img = np.concatenate((img_resized1, roi, img_resized1), axis=1)
    img = cv2.resize(img, (image.shape[1], image.shape[0]))


    result = cv2.addWeighted(image, 1, img, 1, 0)
    
    return result
if __name__ == '__main__':
    model = load_model('full_CNN_model.h5')
    lanes = Lanes()

    # Where to save the output video
    vid_output = '../output2/5.mp4'

    # Location of the input video
    clip1 = VideoFileClip("../output/exp5/5.mp4")

    vid_clip = clip1.fl_image(road_lines)
    vid_clip.write_videofile(vid_output, audio=False)
