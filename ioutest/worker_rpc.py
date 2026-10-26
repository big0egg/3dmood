"""Private trusted child IPC over anonymous pipes, without network sockets."""
import os
import pickle
import select
import struct
import subprocess


class PipeConnection:
    def __init__(self, read_fd, write_fd, timeout=1200):
        self.read_fd = read_fd
        self.write_fd = write_fd
        self.timeout = timeout
        os.set_blocking(write_fd, False)

    def recv(self, count):
        if not select.select([self.read_fd], [], [], self.timeout)[0]:
            raise TimeoutError('等待前端管道超时')
        return os.read(self.read_fd, count)

    def sendall(self, payload):
        payload = memoryview(payload)
        while payload:
            if not select.select([], [self.write_fd], [], self.timeout)[1]:
                raise TimeoutError('写入前端管道超时')
            try:
                count = os.write(self.write_fd, payload[:65536])
            except BlockingIOError:
                continue
            payload = payload[count:]

    def close(self):
        for fd in (self.read_fd, self.write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def pipe_pair():
    parent_read, child_write = os.pipe()
    child_read, parent_write = os.pipe()
    return PipeConnection(parent_read, parent_write), PipeConnection(child_read, child_write)


def _read(connection, count):
    chunks = []
    while count:
        chunk = connection.recv(min(count, 8 * 1024 * 1024))
        if not chunk:
            raise RuntimeError('前端进程断开；检查 worker.log')
        chunks.append(chunk)
        count -= len(chunk)
    return b''.join(chunks)


def receive(connection):
    size = struct.unpack('!Q', _read(connection, 8))[0]
    if size > 1024**3:
        raise ValueError('内部消息超出 1 GiB')
    return pickle.loads(_read(connection, size))


def send(connection, value):
    payload = pickle.dumps(value, protocol=5)
    connection.sendall(struct.pack('!Q', len(payload)) + payload)


class FrontendClient:
    def __init__(self, python, config, track, classes, execution):
        parent, child = pipe_pair()
        self.connection = parent
        try:
            self.process = subprocess.Popen(
                [python, '-B', '-m', 'ioutest.frontend_worker', str(child.read_fd), str(child.write_fd)],
                pass_fds=(child.read_fd, child.write_fd),
            )
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        try:
            send(parent, dict(config=config, track=track, classes=classes, execution=execution))
            response = receive(parent)
            if not response['ok']:
                raise RuntimeError(response['error'])
            self.identity = response['identity']
        except BaseException:
            self.close()
            raise

    def predict(self, rgb, intrinsics, prompts, depth=None):
        send(self.connection, dict(rgb=rgb, intrinsics=intrinsics, prompts=prompts, depth=depth))
        response = receive(self.connection)
        if not response['ok']:
            raise RuntimeError('前端推理失败：' + response['error'])
        return response['result']

    def close(self):
        if self.process.poll() is None:
            try:
                send(self.connection, {'close': True})
            except OSError:
                pass
            self.connection.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        else:
            self.connection.close()
