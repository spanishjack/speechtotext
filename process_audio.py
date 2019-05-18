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
