# %%
import gzip
import os
import socket
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import reduce, partial
from operator import mul
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

while 'src' in os.getcwd():
    os.chdir('..')

# %%
DataSpec = dict[str, any]
DataSpec = dict[str, any]
Logo = list[str]
Host = str
Port = int
Address = tuple[Host, int]
Socket = socket.socket
URL = dict[str, str]
HTTPRequestLine = dict[str, any]
HTTPStatusLine = dict[str, any]
HTTPHeaders = dict[str, any]
HTTPRequest = dict[str, any]
HTTPResponse = list[bytes]

# %%
LOGO = [
    " _  _  _  _  __ _ ",
    "( \\/ )( \\/ )(  ( \\",
    " )  (  )  / /    /",
    "(_/\\_)(__/  \\_)__)",
]

HEADER_TYPES = {
    'Content-Length': int
}

MNIST_TRAIN_INPUTS_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_TARGETS_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_INPUTS_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_TARGETS_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'


# %%

# TODO: add type hints

def first(xs: Sequence[any]) -> Sequence[any]:
    return xs[0]

def second(xs: Sequence[any]) -> Sequence[any]:
    return xs[1]

def last(xs: Sequence[any]) -> Sequence[any]:
    return xs[-1]

def rest(xs: Sequence[any]) -> Sequence[any]:
    return xs[1:]

def take_last(n: int, xs: Sequence[any]):
    return xs[-n:]

def lines(s: str) -> list[str]:
    return s.split("\n")

def unlines(xs: Iterable[any]) -> str:
    return "\n".join(xs)

def unwords(xs: Iterable[any]) -> int:
    return " ".join(xs)

# %%
def list_ndims(xs: list[any]) -> int:
    def go(xs) -> int:
        match xs:
            case []:
                return 1
            case [x, *xs] if not isinstance(x, list):
                return 1
            case [x, *xs] if isinstance(x, list):
                return 1 + go(x)
    return go(xs)

# %%
def list_show_by(delim: str, xs: list):
    assert isinstance(xs, list)

    n = list_ndims(xs)
    if n == 1:
        return delim.join(map(str, xs))
    elif n == 2:
        return unlines(map(lambda x: list_show_by(delim, x), xs))
    else:
        raise Exception(f"Expected 1 or 2 dimensions, found {n}")


# %%
list_show = partial(list_show_by, " ")

# %%
# TODO: make recursive
def dict_show_by(kv_delim: str, entry_delim: str, mp):
    return entry_delim.join(str(k) + kv_delim + str(v) for k, v in mp.items())

dict_show = partial(dict_show_by, "\n", "\n")

# %%
def bits_show(x: int) -> str:
    return format(x, 'b')

# %%

def bytes_lines(bs: bytes) -> list[bytes]:
    return bs.split(b'\n')

def bytes_unlines(lines: list[bytes]) -> bytes:
    return b'\n'.join(lines)

def bytes_words(bs: bytes) -> list[bytes]:
    return bs.split(b' ')

def bytes_unwords(words: list[bytes]) -> bytes:
    return b' '.join(words)

def bytes_show(bs: bytes) -> str:
    return bs.decode()

def bytes_read(s: str) -> bytes:
    return s.encode()

# %%

# TODO: shorten this
def body_show(bs: bytes) -> str:
    match bytes_lines(bs):
        case [b]:
            return bytes_show(b)
        case [fst, snd]:
            return bytes_show(fst) + "\n" + bytes_show(snd)
        case [fst, snd, lst]:
            return (bytes_show(fst) + "\n" +
                    bytes_show(snd) + "\n" +
                    bytes_show(lst))
        case [fst, snd, *_, lst]:
            return (bytes_show(fst) + "\n" +
                    bytes_show(snd) + "\n" +
                    "...\n" +
                    bytes_show(lst))
        case _:
            raise ValueError("Unexpected number of lines in body")

# %%

# HTTPRequestLine constructor

def make_hrqln(meth: str, uri: str, version: str) -> HTTPRequestLine:
    return {
        'method': meth,
        'uri': uri,
        'version': version,
        'type': 'RequestLine'
    }

# HTTPRequestLine selectors

def hrqln_method(rl: HTTPRequestLine) -> str:
    return rl['method']

def hrqln_uri(rl: HTTPRequestLine) -> str:
    return rl['uri']

def hrqln_version(rl: HTTPRequestLine) -> str:
    return rl['version']

# HTTPRequestLine representation invariant/predicate

def is_hrqln(rl: HTTPRequestLine):
    return rl['type'] == 'RequestLine'

# HTTPRequestLine operations

def hrqln_show(rl: HTTPRequestLine) -> str:
    return hrqln_method(rl) + " " + hrqln_uri(rl) + " "  + hrqln_version(rl) + "\r\n"

def hrqln_read(s: str) -> HTTPRequestLine:
    method, uri, version = s.strip().split()
    return make_hrqln(method, uri, version)

# %%

# TODO: fix these types

# HTTPHeaders constructor

def make_hhdrs(hs: dict[str, any]) -> HTTPHeaders:
    return {**{f: hhdrs_lookup_type(f)(v) for f, v in hs.items()}, 'type': 'HTTPHeaders'}

# HTTPHeaders selectors

def hhdrs_fields(hs: HTTPHeaders) -> list[str]:
    return [k for k in hs.keys() if k != 'type']

def hhdrs_lookup(hs: HTTPHeaders, field: str) -> str:
    assert field in hhdrs_fields(hs), f"key '{field}' not found in headers"
    return hs[field]

def hhdrs_lookup_type(field: str) -> type:
    return HEADER_TYPES.get(field, str)

def hhdrs_values(hs: HTTPHeaders) -> list[str]:
    return [hhdrs_lookup(hs, f) for f in hhrds_fields(hs)]

def hhdrs_items(hs: HTTPHeaders) -> list[tuple[str, str]]:
    return [(f, hhdrs_lookup(hs, f)) for f in hhdrs_fields(hs)]

# HTTPHeaders representation invariant/predicate

def is_hhdrs(hs: HTTPHeaders):
   return hs['type'] == 'HTTPHeaders'

# HTTPHeaders operations

def hhdrs_show(hs: HTTPHeaders) -> str:
    return ''.join(str(f) + ": " + str(v) + "\r\n" for f, v in hhdrs_items(hs))

def hhdrs_read(s: str) -> HTTPHeaders:
    return make_hhdrs(dict(h.split(": ") for h in s.strip().split("\r\n")))

def hhdrs_print(hs: HTTPHeaders):
    print(hhdrs_show(hs))

# %%

# HTTPRequest constructor

def make_hrq(rqln: HTTPRequestLine, hs: HTTPHeaders) -> HTTPRequest:
    return {
        'request_line': rqln,
        'headers': hs,
        'type': 'HTTPRequest'
    }

# HTTPRequest selectors

def hrq_rqln(rq: HTTPRequest) -> HTTPRequestLine:
    return rq['request_line']

def hrq_hdrs(rq: HTTPRequest) -> HTTPHeaders:
    return rq['headers']

# HTTPRequest representation invariant/predicate

def is_hrq(rq: HTTPRequest):
    return rq['type'] == 'HTTPRequest'

# HTTPRequest operations

def hrq_show(rq: HTTPRequest) -> str:
    return (hrqln_show(hrq_rqln(rq)) +
            hhdrs_show(hrq_hdrs(rq)) +
            "\r\n")

def hrq_read(s: str) -> HTTPRequest:
    s_rqln, s_hdrs = s.strip().split("\r\n", 1)
    return make_hrq(hrqln_read(s_rqln), hhdrs_read(s_hdrs))

def hrq_print(rq: HTTPRequest):
    print(hrq_show(rq))

def hrq_send(req: HTTPRequest) -> HTTPResponse:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# %%

# HTTPStatusLine constructor

def make_hstln(version: str, status: str, reason: Optional[str]) -> HTTPStatusLine:
    return {
        'version': version,
        'status': status,
        'reason': reason,
        'type': 'HTTPStatusLine'
    }

# HTTPStatusLine selectors

def hstln_version(sl: HTTPStatusLine) -> str:
    return sl['version']

def hstln_status(sl: HTTPStatusLine) -> str:
    return sl['status']

def hstln_reason(sl: HTTPStatusLine) -> Optional[str]:
    return sl['reason']

# HTTPStatusLine representation invariant/predicate

def is_hstln(sl: HTTPStatusLine):
    return sl['type'] == 'HTTPStatusLine'

# HTTPStatusLine operations

def hstln_show(sl: HTTPStatusLine) -> str:
    return hstln_version(sl) + " " + hstln_status(sl) + " " + hstln_reason(sl) + "\r\n"

def hstln_read(s: str) -> HTTPStatusLine:
    version, status, reason = s.strip().split()
    return make_hstln(version, status, reason)

def hstln_print(sl: HTTPStatusLine):
    print(hstln_show(sl))

# %%

# HTTPResponse constructor

def make_hrs(stln: HTTPStatusLine, hs: HTTPHeaders, body: bytes) -> HTTPResponse:
   return {
       'status_line': stln,
       'headers': hs,
       'body': body,
       'type': 'HTTPResponse'
    }

# HTTPResponse selectors

def hrs_stln(rs: HTTPResponse) -> HTTPStatusLine:
    return rs['status_line']

def hrs_hdrs(rs: HTTPResponse) -> HTTPHeaders:
    return rs['headers']

def hrs_body(rs: HTTPResponse) -> bytes:
    return rs['body']

# HTTPResponse representation invariant

def is_hrs(rs: HTTPResponse):
    return rs['type'] == 'HTTPResponse'

# HTTPResponse operations

def hrs_show(rs: HTTPResponse) -> str:
    return (hstln_show(hrs_stln(rs)) +
            hhdrs_show(hrs_hdrs(rs)) +
            "\r\n" +
            bytes_show(hrs_body(rs)))

def hrs_read(s: str) -> HTTPResponse:
    stln_line, lines = s.strip().split("\r\n", 1)
    hdrs_lines, body_lines = lines.rsplit("\r\n\r\n", 1)
    return make_hrs(hstln_read(stln_line), hhdrs_parse(hdrs_lines), bytes_parse(body_lines))

# %%

# Socket constructor

# TODO: add types
@contextmanager
def make_sock() -> Socket:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   try:
       yield s
   finally:
       s.shutdown(socket.SHUT_WR)
       s.close()

# Socket representation invariant

def check_sock(s: Socket):
    raise NotImplementedError()

# Socket operations

def sock_connect(s: Socket, a: Address):
    s.connect(a)

def sock_send(s: Socket, bs: bytes):
   s.sendall(bs)

def sock_recv(s: Socket, n: int) -> bytes:
    acc = b''
    while len(acc) < n:
        chunk = s.recv(n - len(acc))
        if not chunk:
            raise ConnectionError("Socket connection broken")
        acc += chunk
    return acc

def sock_recv1(s: Socket) -> bytes:
    return sock_recv(s, 1)

def sock_recv_while(s: Socket, pred: Callable[list[bytes], bool]) -> bytes:
    acc = []
    while pred(acc):
        acc.append(sock_recv1(s))
    return b''.join(acc)

def sock_recv_match(s: Socket, bs: list[bytes]) -> bytes:
    assert all(len(b) == 1 for b in bs)
    return sock_recv_while(s, lambda acc: take_last(len(bs), acc) != bs)

def sock_recvln(s: Socket) -> bytes:
    return sock_recv_match(s, [b'\n'])

def sock_recv_lines(s: Socket, n: int) -> list[bytes]:
    acc = []
    while len(acc) < n:
        acc.append(sock_recvln(s))
    return acc

# High-level socket operations

# Receive HTTP headers
def sock_recv_hhdrs(s: Socket) -> bytes:
    acc = []
    while not acc or last(acc) != b"\r\n":
       acc.append(sock_recvln(s))
    return b''.join(acc)

def sock_recv_hrs(s: Socket) -> tuple[HTTPStatusLine, HTTPHeaders, bytes]:
    stln = hstln_read(sock_recvln(s).decode())
    hdrs = hhdrs_read(sock_recv_hhdrs(s).decode())
    body = sock_recv(s, hhdrs_lookup(hdrs, 'Content-Length'))
    return stln, hdrs, body

# %%

def make_url(scheme: str, host: str, resource: str) -> URL:
    return {
        'scheme': scheme,
        'host': host,
        'resource': resource,
        'type': 'URL'
    }

def url_scheme(url: URL) -> str:
    return url['scheme']

def url_host(url: URL) -> str:
    return url['host']

def url_resource(url: URL) -> str:
    return url['resource']

# URL representation invariant

def is_url(url: URL) -> bool:
    return url['type'] == 'URL'

# URL operations

def url_show(url: URL) -> str:
    return url_scheme(url) + "://" + url_host(url) +  "/" + url_resource(url)

def url_print(url: URL):
    print(url_show(url))

def url_read(s: str) -> URL:
    match s.strip().split("://", 1):
        case [rest]:
            return url_read("://" + rest)
        case [scheme, rest]:
            match rest.split("/", 1):
                case [host, resource]:
                    return make_url(scheme, host, resource)
                case [host]:
                    return make_url(scheme, host, '/')
                case _:
                    raise ValueError("Invalid URL format")
    raise ValueError("Invalid URL format")

url = make_url('', 'example.com', 'path/to/resource')

def url_fetch(url: URL) -> bytes:
    host = url_host(url)
    uri = url_resource(url)

    with make_sock() as s:
        sock_connect(s, (host, 80))
        rq = make_hrq(make_hrqln('GET', uri, 'HTTP/1.1'),
                      make_hhdrs({'Host': host,
                                  'User-Agent': 'xyn/0.1',
                                  'Accept': '*/*'}))
        sock_send(s, hrq_show(rq).encode())

        stln, hdrs, body = sock_recv_hrs(s)
    return body

def fetch(s: str) -> bytes:
    url = url_read(s)
    if not is_url(url):
        raise ValueError("Expected a URL")
    return url_fetch(url)

# %%

if 'mnist-train-inputs.gz' not in os.listdir('data'):
    with open('data/mnist-train-inputs.gz', 'xb') as f:
        f.write(fetch(MNIST_TRAIN_INPUTS_URL))

if 'mnist-train-targets.gz' not in os.listdir('data'):
    with open('data/mnist-train-targets.gz', 'xb') as f:
        f.write(fetch(MNIST_TRAIN_TARGETS_URL))

if 'mnist-test-inputs.gz' not in os.listdir('data'):
    with open('data/mnist-test-inputs.gz', 'xb') as f:
        f.write(fetch(MNIST_TEST_INPUTS_URL))

if 'mnist-test-targets.gz' not in os.listdir('data'):
    with open('data/mnist-test-targets.gz', 'xb') as f:
        f.write(fetch(MNIST_TEST_TARGETS_URL))

# %%

def make_network_dspec(url: str, path: str, file_type: str, n_samples: int,
                       shape: list[int], n_header: int) -> DataSpec:
    return {
        'data': (url, path),
        'tag': 'network',
        'file_type': file_type,
        'n_samples': n_samples,
        'shape': shape,
        'size': reduce(mul, shape, 1),
        'n_header': n_header,
        'type': 'DataSpec'
    }

# TODO: add file_type checking
def make_disk_dspec(path: str, file_type: str, n_samples: int,
                   shape: list[int], n_header: int) -> DataSpec:
    assert os.path.exists(path), f"File '{path}' does not exist"
    return {
        'data': path,
        'tag': 'disk',
        'file_type': file_type,
        'n_samples': n_samples,
        'shape': shape,
        'size': reduce(mul, shape, 1),
        'n_header': n_header,
        'type': 'DataSpec'
    }

def make_np_dspec(data: np.ndarray, n_samples: int, shape: list[int]) -> DataSpec:
    assert len(data) == n_samples, f"Expected {n_samples} samples, got {len(data)}"
    assert rest(list(data.shape)) == shape, f"Expected shape {shape}, got {data.shape}"
    return {
        'data': data,
        'tag': 'numpy',
        'n_samples': n_samples,
        'shape': shape,
        'size': reduce(mul, shape, 1)
    }

# DataSpec selectors

def dspec_data(ds: DataSpec) -> str:
    return ds['data']

def dspec_tag(ds: DataSpec) -> str:
    return ds['tag']

def dspec_is_network(ds: DataSpec) -> bool:
    return dspec_tag(ds) == 'network'

def dspec_is_disk(ds: DataSpec) -> bool:
    return dspec_tag(ds) == 'disk'

def dspec_is_np(ds: DataSpec) -> bool:
    return dspec_tag(ds) == 'numpy'

def dspec_is_filelike(ds: DataSpec) -> bool:
    return dspec_is_network(ds) or dspec_is_disk(ds)

def dspec_is_primitive(ds: DataSpec) -> bool:
    return dspec_is_np(ds)

def dspec_is_compound(ds: DataSpec) -> bool:
    return dspec_is_filelike(ds)

def dspec_file_type(ds: DataSpec) -> str:
    assert dspec_is_filelike(ds), "DataSpec is not a file-like type"
    return ds['file_type']

def dspec_n_header(fs: DataSpec) -> int:
    assert dspec_is_filelike(fs), "DataSpec is not a file-like type"
    return fs['n_header']

def dspec_n_samples(ds: DataSpec) -> int:
    return ds['n_samples']

def dspec_shape(ds: DataSpec) -> list[int]:
    return ds['shape']

def dspec_size(ds: DataSpec) -> int:
    return ds['size']

# DataSpec representation invariant

def is_dspec(fs: DataSpec) -> bool:
    return fs['type'] == 'DataSpec'

# DataSpec operations

def dspec_eval(ds: DataSpec) -> DataSpec:
    if dspec_is_primitive(ds):
        return ds
    elif dspec_is_disk(ds):
        match dspec_file_type(ds):
            case 'gzip':
                with gzip.open(dspec_data(ds), 'rb') as f:
                    f.read(dspec_n_header(ds))
                    buf = f.read(dspec_size(ds) * dspec_n_samples(ds))
                    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                    data = data.reshape(dspec_n_samples(ds), *dspec_shape(ds))
                    return dspec_eval(make_np_dspec(data, dspec_n_samples(ds), dspec_shape(ds)))
            case _:
                raise NotImplementedError(f"File type '{dspec_file_type(ds)}' not supported")
    elif dspec_is_network(ds):
        url, path = dspec_data(ds)
        if os.path.exists(path):
            return dspec_eval(make_disk_dspec(path,
                                              dspec_file_type(ds),
                                              dspec_n_samples(ds),
                                              dspec_shape(ds),
                                              dspec_n_header(ds)))
        else:
            body = fetch(url)
            with open(path, 'xb') as f:
                f.write(body)
            return dspec_eval(make_disk_dspec(path,
                                              dspec_file_type(ds),
                                              dspec_n_samples(ds),
                                              dspec_shape(ds),
                                              dspec_n_header(ds)))
    else:
        raise NotImplementedError(f"DataSpec type '{dspec_tag(ds)}' not supported")


# %%
def logo_show(l: Logo) -> str:
    return unlines(l)

def logo_read(s: str) -> Logo:
    return lines(s)

def logo_print(l: Logo):
    print(logo_show(l))

# %%
def make_regression_dataset(config, key):
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_outputs = config["n_outputs"]
    noise_scale = config["noise_scale"]

    key, subkey = mx.random.split(key)
    true_weights = mx.random.normal([n_features, n_outputs], key=subkey)

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples, n_outputs], key=subkey)

    targets = mx.matmul(inputs, true_weights) + noise

    aux = {"weights": true_weights.tolist(), "noise": noise.tolist()}
    return mx.concatenate([inputs, targets], axis=1), aux


# %%
def make_sum_dataset(config, key):
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_outputs = config["n_outputs"]
    noise_scale = config["noise_scale"]

    assert n_outputs == 1, "sum dataset must have exactly 1 output"

    true_weights = mx.ones([n_features, n_outputs])

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples, n_outputs], key=subkey)

    targets = mx.matmul(inputs, true_weights) + noise

    aux = {"weights": true_weights.tolist(), "noise": noise.tolist()}
    return mx.concatenate([inputs, targets], axis=1), aux


# %%
def make_mixture_dataset(config, key):
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_outputs = config["n_outputs"]
    weights = mx.array(config["weights"])
    means = mx.array(config["means"])
    covs = mx.array(config["covs"])

    assert n_features == 2, "mixture dataset must have exactly 2 features"
    assert n_outputs == 1, "mixture dataset must have exactly 1 output"

    key, subkey = mx.random.split(key)
    targets = mx.random.categorical(weights, num_samples=n_samples, key=key).tolist()

    key, *subkeys = mx.random.split(key, n_samples + 1)
    inputs = mx.array(
        [
            mx.random.multivariate_normal(means[i], covs[i], key=subkey, stream=mx.cpu)
            for i, subkey in zip(targets, subkeys)
        ]
    )
    targets = mx.array(targets)[:, None]

    aux = {
        "weights": weights.tolist(),
        "means": means.tolist(),
        # "covs": covs.tolist()
    }
    return mx.concatenate([inputs, targets], axis=1), aux

# %%
def show_ix(ix):
    match ix:
        case int(i):
            return str(i)
        case (int(i), int(j)) | [int(i), int(j)]:
            return f"{i}-{j}"
        case _:
            raise NotImplementedError("show_ix can only be applied to integers and pairs of integers")

# %%
def show_header(feat_ixs, out_ixs):
    return unwords(["x" + show_ix(i) for i in feat_ixs] + ["y" + show_ix(i) for i in out_ixs])

# %%
def main() -> None:
    readr = ArgumentParser(
        prog="xyn",
        description="xyn generates xynthetic data",
        epilog=unlines(LOGO),
        formatter_class=RawDescriptionHelpFormatter,
    )
    subreadrs = parser.add_subparsers(required=True)

    readr_regression = subparsers.add_parser("reg")
    readr_regression.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    readr_regression.add_argument(
        "-n",
        "--noise",
        type=float,
        default=1e-2,
        help="Amount of gaussian noise used to generate data",
    )
    readr_regression.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                        generate the dataset",
    )
    readr_regression.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    readr_regression.add_argument(
        "n_features",
        type=int,
        help="Number of input features \
                        to generate data",
    )
    readr_regression.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs = '?',
        help="Number of output variables for each data sample"
    )
    readr_regression.set_defaults(func=make_regression_dataset)

    readr_sum = subparsers.add_parser("sum")
    readr_sum.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    readr_sum.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0,
        help="Amount of gaussian noise used to generate data",
    )
    readr_sum.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                        generate the dataset",
    )
    readr_sum.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    readr_sum.add_argument(
        "n_features",
        type=int,
        help="Number of input features \
                        to generate data",
    )
    readr_sum.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs='?',
        help="Number of output variables for each data sample",
    )
    readr_sum.set_defaults(func=make_sum_dataset)

    readr_mixture = subparsers.add_parser("mix")
    readr_mixture.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    readr_mixture.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0,
        help="Amount of gaussian noise used to generate data",
    )
    readr_mixture.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                                generate the dataset",
    )
    readr_mixture.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    readr_mixture.add_argument(
        "n_features",
        type=int,
        default=2,
        nargs='?',
        help="Number of input features \
                        to generate data",
    )
    readr_mixture.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs='?',
        help="Number of output variables for each data sample",
    )
    readr_mixture.set_defaults(func=make_mixture_dataset)

    args = readr.parse_args()

    seed = args.seed

    config = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_outputs": args.n_outputs,
        "noise_scale": args.noise,
        "weights": [0.3, 0.4, 0.3],
        "means": [[-3, -2], [0, 0], [3, 2]],
        "covs": [[[1, 0.5], [0.5, 1]], [[1, -0.7], [-0.7, 1]], [[1, 0.3], [0.3, 1]]],
    }
    dataset, aux = args.func(config, mx.random.key(seed))
    if args.all:
        print(dict_show({k: list_show(v) for k, v in aux.items()}))
    print(show_header(range(1, args.n_features + 1), range(1, args.n_outputs + 1)))
    print(list_show(dataset.tolist()))


# TODO:
# - [x] enhance: remove delim argument
# - [x] feat: implement verbose mode
# - [x] enhance: rename verbose to all
# - [ ] enhance: improve --help documentation for subcommands
# - [ ] enhance: improve README documentation for subcommands
# - [x] enhance: remove n_outputs option for `xyn sum` subcommand
# - [x] feat: implement `xyn reg` subcommand
# - [x] feat: implement sum dataset
# - [ ] feat: implement header support for multidimensional data (x1-j)
# - [ ] fix: sum dataset be exact sum (I think matmul is breaking it), but is inexact
# - [x] feat: implement `xyn sum` subcommand
# - [x] feat: implement gaussian mixture dataset
# - [ ] feat: implement mnist test set
# - [ ] feat: implement simple dataset combinators
# - [ ] feat: implement multiple input variables in mixture dataset
# - [ ] feat: implement covariance printing in mixture dataset
# - [ ] enhance: refactor redundant readr arguments for mixture dataset
# - [ ] feat: implement sum dataset for integral valued inputs
# - [ ] feat: implement real-valued binary classification dataset
# - [ ] feat: implement option to display header
# - [ ] feat: implement mnist dataset generation
# - [ ] feat: implement old faithful dataset generation
# - [ ] fix: array values should always print as decimals, but sometimes print in scientific notation
# - [x] fix: fix emacs/magit git setup
# - [x] feat: improve readme by showing how to use `xyn` with other unix commands
# - [ ] enhance: Add examples to `-h` help
# - [ ] enhance: improve typechecking of list_ndims
# - [ ] enhance: improve function typechecking and precondition checking
# - [ ] enhance: implement unit and expect tests
# - [ ] enhance: implement property-based testing
# - [x] feat: multiple output variables
# - [ ] enhance: update readme for multiple output variables
# - [ ] feat: implement multi-dimensional classification dataset
# - [x] feat: implement splitting into train/test data (done by datasplit program)
# - [ ] enhance: refactor make_regression_dataset, make_sum_dataset
# - [ ] enhance: refactor readr_regression, parser_sum repeated code
