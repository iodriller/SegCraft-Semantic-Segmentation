import os
from youtube_download import youtube_download
from separate_to_jpgs import separate_to_jpgs
from semantic_images import semantic_images
from combine_and_make_video import combine_and_make_video


class process_and_run():
    def __init__(self, dirname='frames', net='deeplabv3_resnet101'):
        self.dirname = dirname
        self.net = net

    def run(self, string):
        # download the youtube video
        print('downloading the video')
        yt = youtube_download().download(string)
        # create directory for background images
        separate_to_jpgs().create_directory()
        # divide video into images
        separate_to_jpgs().separate(yt)
        # create the model do the segmentation
        print('Running the net')
        model = semantic_images(self.net).create_model()
        directory = os.fsencode(self.dirname)
        print(f"need to iterate through {len(os.listdir(directory))} images. "
              f"If taking too long, consider adjusting parameters, i.e. length of the video")
        # create directory for overlay images
        semantic_images(self.net).create_directory()
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            print('maindayim',self.net)
            semantic_images(self.net).create_semantic_images(self.dirname, filename, model)
        # create directory for combined images
        combine_and_make_video().create_directory()
        # combine the background and the overlay
        combine_and_make_video().combine_background_and_overlay()
        # make a video from the combined images
        print('Preparing the final video')
        combine_and_make_video().make_video_from_jpgs()
