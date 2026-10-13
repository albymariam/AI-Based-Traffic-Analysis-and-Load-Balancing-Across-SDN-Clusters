"""Mice-flow tool, stdlib only; runs inside Mininet hosts.
  server: python3 mouse.py server 6000
  client: python3 mouse.py client <ip> <port> <bytes>   -> prints one JSON line (FCT measured here)"""
import json
import socket
import sys
import threading
import time


def _handle(c):
    try:
        while c.recv(65536):
            pass
        c.sendall(b"A")
    except OSError:
        pass
    finally:
        c.close()


def server(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(512)
    while True:
        c, _ = s.accept()
        threading.Thread(target=_handle, args=(c,), daemon=True).start()


def client(ip, port, nbytes):
    buf, t0 = b"x" * 65536, time.time()
    try:
        s = socket.create_connection((ip, port), timeout=15)
        left = nbytes
        while left > 0:
            n = min(left, len(buf))
            s.sendall(buf[:n])
            left -= n
        s.shutdown(socket.SHUT_WR)
        ok = s.recv(1) == b"A"
        s.close()
        print(json.dumps({"ok": ok, "fct_s": time.time() - t0, "bytes": nbytes}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:60]}))


if __name__ == "__main__":
    if sys.argv[1] == "server":
        server(int(sys.argv[2]))
    else:
        client(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))