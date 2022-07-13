import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
from argparse import ArgumentParser


"""
prepare symlinks and serve webpage
uses code from https://stackoverflow.com/questions/27977972/how-do-i-setup-a-local-http-server-using-python
"""

output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "output"))
out_dir = sorted(os.listdir(output_folder))[-1]

ap = ArgumentParser()
ap.add_argument("--dataset_root", default="../input/datasets/cityscapes/leftImg8bit/val")
ap.add_argument("--eval_folder", default=out_dir)
args = ap.parse_args()

try:
    os.remove("eval")
except:
    pass

os.symlink(os.path.join(output_folder, out_dir, "raw"), "eval")


def create_index():
    dir = os.listdir("eval")
    with open("eval/index.txt", "w+") as fp:
        fp.write("\n".join(filter(lambda x: ".csv" not in x and ".txt" not in x, dir)))


try:
    os.remove("dataset")
except Exception:
    pass

os.symlink(args.dataset_root, "dataset")


create_index()


class Serv(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/error_inf.html'
        if self.path[-4:] != ".png":
            try:
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)
            except:
                file_to_open = "File not found!"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))
        else:
            try:
                file_to_open = open(self.path[1:], "rb").read()
                self.send_response(200)
            except:
                file_to_open = "File not found"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(file_to_open)


webbrowser.open("http://localhost:8080")
httpd = HTTPServer(('localhost', 8080), Serv)
httpd.serve_forever()
