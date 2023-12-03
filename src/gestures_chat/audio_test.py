from pydub import AudioSegment
from pydub.playback import play
import threading
import time

if __name__ == "__main__":
    sound = AudioSegment.from_wav("audio/iloveu2.wav")
    threading.Thread(target=play, args=(sound,), daemon=True).start()

    time.sleep(0.35)
    threading.Thread(target=play, args=(sound,), daemon=True).start()

    print("Done")
    time.sleep(5)
