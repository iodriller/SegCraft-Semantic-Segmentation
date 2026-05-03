import cv2
import os


class combine_and_make_video():
    def __init__(self, path_frames='frames', path_overlay='overlay', combined='combined', jpg_qualty=450,
                 video_name='video.avi', fps=1):
        self.path_overlay = path_overlay
        self.path_frames = path_frames
        self.path_overlay = path_overlay
        self.combined = combined
        self.jpg_qualty = jpg_qualty
        self.video_name = video_name
        self.fps = fps

    def create_directory(self):
        if not os.path.exists(self.combined):
            os.mkdir(self.combined)

    def combine_background_and_overlay(self):
        dir_background = os.fsencode(self.path_frames)
        dir_overlay = os.fsencode(self.path_overlay)
        for bckg, ovrl in zip(os.listdir(dir_background), os.listdir(dir_overlay)):
            bckg, ovrl = bckg.decode("utf-8"), ovrl.decode("utf-8")
            background = cv2.imread(self.path_frames + '/' + bckg)
            overlay = cv2.imread(self.path_overlay + '/' + ovrl)
            # make sure the shapes are matching, otherwise won't combine
            # print(np.shape(background), np.shape(overlay))
            added_image = cv2.addWeighted(background, 0.4, overlay, 0.1, 0)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_qualty]
            large = cv2.resize(added_image, (0, 0), fx=3.0, fy=3.0)
            cv2.imwrite(self.combined + '/' + bckg + '_' + self.combined + '.jpg', large, encode_param)

    def make_video_from_jpgs(self):
        images = [img for img in os.listdir(self.combined) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(self.combined, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(self.video_name, 0, self.fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.combined, image)))

        cv2.destroyAllWindows()
        video.release()
