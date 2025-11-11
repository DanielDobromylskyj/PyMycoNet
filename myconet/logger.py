import inspect
import atexit
import time


@atexit.register
def close_log():
    with open("logs/latest.txt", "rb") as f:
        data = f.read()

    with open(f"logs/log_{round(time.time())}.txt", "wb") as f:
        f.write(data)


class Logger:
    def __init__(self):
        self.path = "logs/latest.txt"

    def log(self, *args, sep=" ", end="\n"):
        passed_string = sep.join(map(str, args)) + end

        frame_info = inspect.stack()[1]
        frame = frame_info.frame

        # Try to get class name (if inside a method)
        cls = None
        if 'self' in frame.f_locals:
            cls = frame.f_locals['self'].__class__.__name__

        func = frame_info.function

        if cls:
            log_string = f"[INFO] {cls}.{func}: {passed_string}"
        else:
            log_string = f"[INFO] {func}: {passed_string}"

        with open(self.path, "a") as f:
            f.write(log_string)

        print(log_string, end="", sep="")
