import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8080
ROOT = "/home/user/video"

class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        if "download=1" in self.path:
            filename = os.path.basename(self.path.split("?", 1)[0]) or "video.mp4"
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isfile(path) and "Range" in self.headers:
            range_header = self.headers["Range"]
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                file_size = os.path.getsize(path)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                if start <= end:
                    self.send_response(206)
                    self.send_header("Content-type", self.guess_type(path))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
                    self.end_headers()
                    f = open(path, "rb")
                    f.seek(start)
                    self._range_len = end - start + 1
                    return f
        self._range_len = None
        return super().send_head()

    def copyfile(self, source, outputfile):
        if getattr(self, "_range_len", None) is not None:
            remaining = self._range_len
            buf_size = 64 * 1024
            while remaining > 0:
                chunk = source.read(min(buf_size, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        else:
            super().copyfile(source, outputfile)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), RangeHTTPRequestHandler)
    print(f"Serving the Mr. Bean comedy short on http://0.0.0.0:{PORT}")
    server.serve_forever()
