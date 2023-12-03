from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_wav("audio/iloveu2.wav")

for _ in range(3):
    play(sound)
