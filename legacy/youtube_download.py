# i.e. download a normal traffic driving video from youtube
from pytube import YouTube

class youtube_download():
    def download(self, string):
        yt = YouTube(string)
        # title of the youtube video
        print('the title of the youtube video is:', yt.title)
        stream = yt.streams.first()
        stream.download()
        return yt

