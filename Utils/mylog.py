import threading
from queue import Queue
import os


class MyLogStream:
    def __init__(self,log_file_path,mode='w'):
        self.log_stream=open(log_file_path,mode=mode)

    def write_line(self,str_line):
        str_line=str(str_line)+'\n'
        self.log_stream.write(str_line)
        self.flush()

    def flush(self):
        if self.log_stream and hasattr(self.log_stream, "flush"):
            self.log_stream.flush()

    def close(self):
        if self.log_stream:
            try:
                self.flush()
            finally:
                stream = self.log_stream
                self.log_stream = None
                if hasattr(stream, "close"):
                    stream.close()