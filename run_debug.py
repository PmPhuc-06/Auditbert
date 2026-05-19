import traceback
import sys

try:
    import app
    with open("error.log", "w", encoding="utf-8") as f:
        f.write("OK\n")
except Exception as e:
    with open("error.log", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
