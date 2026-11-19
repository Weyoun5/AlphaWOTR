# A static server for the worker check (test.html): serves this folder, takes the page's POST /result into
# result.json and exits.   python3 serve.py PORT
import http.server, sys, os
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm'}
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        open('result.json', 'wb').write(self.rfile.read(n))
        self.send_response(200); self.end_headers()
        os._exit(0)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
