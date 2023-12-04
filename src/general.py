from detect_pattern.authenticate import authenticate
from gestures_chat.aruco_rpi import PatternInterpreter
from detect_pattern.detect import Figure, get_picam2

if __name__ == "__main__":
    valid_figures = [
        Figure("quadrilateral", "red", (50, 40, 140), 4),
        Figure("quadrilateral", "yellow", (35, 155, 172), 4),
        Figure("triangle", "blue", (140, 85, 45), 3),
        Figure("pentagon", "green", (50, 85, 55), 5),
        # Figure("triangle", "red", (0, 0, 178), 3),
        # Figure("triangle", "yellow", (0, 145, 200), 3),
        # Figure("quadrilateral", "green", (30, 125, 15), 4),
        # Figure("pentagon", "blue", (249, 54, 0), 5),
    ]

    password = [
        valid_figures[0],
        valid_figures[1],
        valid_figures[2],
        valid_figures[3],
    ]

    picam2 = get_picam2()

    # authenticate(picam2, password, valid_figures)
    pattern_interpreter = PatternInterpreter(picam2)
    pattern_interpreter.chat(aruco_target_id=0)
