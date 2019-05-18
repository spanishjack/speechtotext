#Get the audio file
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=ouRbkBAOGEw'])

#process the audio file
import pydub
from pydub import AudioSegment

#directory path to the mp3 source file
dirpath='/path_to_your_directory/'
filename='JFK - We choose to go to the Moon, full length-ouRbkBAOGEw'
inputformat='mp3'
outputformat='wav'

#initialize our audio segment and set channels to mono
audio = AudioSegment.from_file(dirpath+filename+'.'+inputformat,format=inputformat)
audio = audio.set_channels(1)

#export audio to wav format
audio.export(dirpath+filename+'.'+outputformat, format=outputformat)
