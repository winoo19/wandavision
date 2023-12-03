from pydub import AudioSegment
from pydub.playback import play
import threading
import time

if __name__ == "__main__":
    sound = AudioSegment.from_wav("audio/iloveu2.wav")
    t1 = threading.Thread(target=play, args=(sound,), daemon=True)
    t1.start()

    time.sleep(0.35)
    t1 = threading.Thread(target=play, args=(sound,), daemon=True)
    t1.start()

    print("Done")
    time.sleep(5)
