from detect_pattern.authenticate import authenticate
from gestures_chat.aruco_rpi import PatternInterpreter


if __name__ == "__main__":
    authenticate()
    pattern_interpreter = PatternInterpreter()
    pattern_interpreter.chat(aruco_target_id=0)
