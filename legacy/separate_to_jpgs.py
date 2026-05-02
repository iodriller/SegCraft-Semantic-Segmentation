import os
import cv2


class separate_to_jpgs():
    def __init__(self, dirname='frames', jpg_qualty=450, frame_rate_limiter=20):
        self.dirname = dirname
        self.jpg_qualty = jpg_qualty
        self.frame_rate_limiter = frame_rate_limiter

    def create_directory(self):
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)

    def separate(self, yt):
        # convert the youtube video to frames under the frames directory
        vidcap = cv2.VideoCapture(yt.title + '.mp4')
        success, image = vidcap.read()

        count = 0
        while success:
            if count % self.frame_rate_limiter == 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_qualty]
                file_name = "%d.jpg" % (count / self.frame_rate_limiter)
                large = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)
                cv2.imwrite(os.path.join(self.dirname, file_name), large, encode_param)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
