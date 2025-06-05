# %%
import gzip
import os
import stat
import socket
import urllib.request
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

while 'src' in os.getcwd():
    os.chdir('..')

# %%
DataSpec = dict[str, any]
Logo = list[str]
Host = str
Port = int
Address = tuple[Host, int]
Socket = socket.socket
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

# %%

# TODO: add type hints

def last(xs: Sequence[any]) -> Sequence[any]:
    return xs[-1]

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

# HTTPRequestLine constructor

def make_httpreqline(meth: str, uri: str, version: str) -> HTTPRequestLine:
    return {
        'method': meth,
        'uri': uri,
        'version': version,
        'type': 'RequestLine'
    }

# HTTPRequestLine selectors

def httpreqline_method(rl: HTTPRequestLine) -> str:
    return rl['method']

def httpreqline_uri(rl: HTTPRequestLine) -> str:
    return rl['uri']

def httpreqline_version(rl: HTTPRequestLine) -> str:
    return rl['version']

# HTTPRequestLine representation invariant

def check_httpreqline(rl: HTTPRequestLine):
    raise NotImplementedError("check_http_request_line is not implemented yet")

# HTTPRequestLine operations

def httpreqline_show(rl: HTTPRequestLine) -> str:
    return httpreqline_method(rl) + " " + httpreqline_uri(rl) + " "  + httpreqline_version(rl) + "\r\n"

def httpreqline_parse(s: str) -> HTTPRequestLine:
    raise NotImplementedError("read_http_request_line is not implemented yet")

req_line = make_httpreqline('GET', '/', 'HTTP/1.1')

# %%

# TODO: fix these types

# # HTTPHeaders constructor

def make_httpheaders(hs: dict[str, str]) -> HTTPHeaders:
    return {**hs, 'type': 'HTTPHeaders'}

# # HTTPHeaders selectors

def httpheaders_fields(hs: HTTPHeaders) -> list[str]:
    return [k for k in hs.keys() if k != 'type']

def httpheaders_lookup(hs: HTTPHeaders, field: str) -> str:
    assert field in httpheaders_fields(hs), f"key '{field}' not found in headers"
    return hs[field]

def httpheaders_values(hs: HTTPHeaders) -> list[str]:
    return [httpheaders_lookup(hs, f) for f in httpheaders_fields(hs)]

def httpheaders_items(hs: HTTPHeaders) -> list[tuple[str, str]]:
    return [(f, httpheaders_lookup(hs, f)) for f in httpheaders_fields(hs)]

# # HTTPHeaders representation invariant

def check_httpheaders(hs: HTTPHeaders):
    raise NotImplementedError("check_http_headers is not implemented yet")

# # HTTPHeaders operations

def httpheaders_show(hs: HTTPHeaders) -> str:
    return ''.join(str(f) + ": " + str(v) + "\r\n" for f, v in httpheaders_items(hs))

def httpheaders_parse(s: str) -> HTTPHeaders:
    return make_httpheaders(dict(h.split(": ") for h in s.strip().split("\r\n")))

hs = make_httpheaders({'Host': 'example.com', 'User-Agent': 'xyn/0.1', 'Accept': '*/*'})

# %%

# HTTPRequest constructor

def make_httpreq(rl: HTTPRequestLine, hs: HTTPHeaders) -> HTTPRequest:
    return {
        'request_line': rl,
        'headers': hs,
        'type': 'HTTPRequest'
    }

# HTTPRequest selectors

def httpreq_reqline(req: HTTPRequest) -> HTTPRequestLine:
    return req['request_line']

def httpreq_headers(req: HTTPRequest) -> HTTPHeaders:
    return req['headers']

# HTTPRequest representation invariant

def check_httpreq(req: HTTPRequest):
    raise NotImplementedError("check_http_request is not implemented yet")

# HTTPRequest operations

def httpreq_show(req: HTTPRequest) -> str:
    return (httpreqline_show(httpreq_reqline(req)) +
            httpheaders_show(httpreq_headers(req)) +
            "\r\n")

def httpreq_parse(s: str) -> HTTPRequest:
    raise NotImplementedError("read_http_request is not implemented yet")

def httpreq_print(req: HTTPRequest):
    print(httpreq_show(req))

def httpreq_send(req: HTTPRequest) -> HTTPResponse:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

req = make_httpreq(req_line, hs)

# req = make_http_request('GET', 'http://example.com', 'HTTP/1.1',
#                   {'Host': 'example.com', 'User-Agent': 'xyn/0.1'})

# req = make_http_request('GET', '/', 'HTTP/1.1',
#                         {'Host': 'example.com', 'User-Agent': 'xyn/0.1', 'Accept': '*/*'})

# %%

def make_httpstatline(version: str, status: str, reason: Optional[str]) -> HTTPStatusLine:
    return {
        'version': version,
        'status': status,
        'reason': reason
        'type': HTTPStatusLine
    }


resp_line = ...

# %%

# Socket constructor

# TODO: add types
@contextmanager
def make_socket() -> Socket:
   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   try:
       yield s
   finally:
       s.shutdown(socket.SHUT_WR)
       s.close()

# Socket representation invariant

def check_socket(s: Socket):
    raise NotImplementedError()

# Socket operations

def socket_connect(s: Socket, a: Address):
    s.connect(a)

def socket_send(s: Socket, bs: bytes):
   s.sendall(bs)

def socket_recv(s: Socket, n: int) -> bytes:
    return s.recv(n)

def socket_recv1(s: Socket) -> bytes:
    return socket_recv(s, 1)

def socket_recv_while(s: Socket, pred: Callable[list[bytes], bool]) -> bytes:
    acc = []
    while pred(acc):
        acc.append(socket_recv1(s))
    return b''.join(acc)

def socket_recv_match(s: Socket, bs: list[bytes]) -> bytes:
    assert all(len(b) == 1 for b in bs)
    return socket_recv_while(s, lambda acc: take_last(len(bs), acc) != bs)

def socket_recvln(s: Socket) -> bytes:
    return socket_recv_match(s, [b'\n'])

def socket_recv_lines(s: Socket, n: int) -> list[bytes]:
    acc = []
    while len(acc) < n:
        acc.append(socket_recvln(s))
    return acc

# High-level socket operations

def socket_recv_httpheaders(s: Socket) -> bytes:
    acc = []
    while not acc or last(acc) != b"\r\n":
       acc.append(socket_recvln(s))
    return b''.join(acc)

def socket_recv_httpresp(s: Socket):
    raise NotImplementedError()

with make_socket() as s:
    socket_connect(s, ("www.example.com", 80))
    socket_send(s, httpreq_show(req).encode())
    # chunk = socket_recv_lines(s, 3)
    chunk = socket_recv_httpheaders(s)
    print(chunk.decode())
    #chunk1 = socket_recv(s, 2048)
    #print(type(chunk1))
    # chunk1 = socket_recv(s, 2048).decode()
    # print(chunk1)
    # print("---")
    # print(chunk2)



# %%
def fetch(url: str) -> list[bytes]:
    ...

# %%
def logo_show(l: Logo) -> str:
    return unlines(l)

def logo_parse(s: str) -> Logo:
    return lines(s)
    # raise NotImplementedError("logo_parse is not implemented yet")

def logo_print(l: Logo):
    print(logo_show(l))

# %%
#  DataSpec constructor

# TODO: refactor FileSpec into a separate data abstraction
def make_dataspec(train_inputs_file, train_targets_file,
                  test_inputs_file, test_targets_file,
                  file_type,
                  n_train, n_test,
                  input_shape, input_size,
                  target_shape, target_size,
                  input_file_header_bufsize, target_file_header_bufsize,
                  origin, extra) -> DataSpec:
    match origin:
        case 'url':
            assert 'url' in extra
            out = {
                'train_inputs_file': train_inputs_file,
                'train_targets_file': train_targets_file,
                'test_inputs_file': test_inputs_file,
                'test_targets_file': test_targets_file,
                'file_type': file_type,
                'n_train': n_train,
                'n_test': n_test,
                'input_shape': input_shape,
                'input_size': input_size,
                'target_shape': target_shape,
                'target_size': target_size,
                'input_file_header_bufsize': input_file_header_bufsize,
                'target_file_header_bufsize': target_file_header_bufsize,
                'origin': origin,
                'extra': extra,
                'type': 'DataSpec'
            }
            return out
        case 'disk':
            raise NotImplementedError()
        case 'synthetic':
            raise NotImplementedError()

MNIST_SPEC = make_dataspec(
    train_inputs_file='train-images-idx3-ubyte.gz',
    train_targets_file='train-labels-idx1-ubyte.gz',
    test_inputs_file='t10k-images-idx3-ubyte.gz',
    test_targets_file='t10k-labels-idx1-ubyte.gz',
    file_type='gzip',
    n_train=60_000,
    n_test=10_000,
    input_shape=[28, 28],
    input_size = 28 * 28,
    target_shape=[1],
    target_size=1,
    input_file_header_bufsize=16,
    target_file_header_bufsize=8,
    origin='url',
    extra={
        'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        }

)

# %%

# DataSpec selectors

def dataspec_train_inputs_file(spec):
    return spec['train_inputs_file']

def dataspec_train_targets_file(spec):
    return spec['train_targets_file']

def dataspec_test_inputs_file(spec):
    return spec['test_inputs_file']

def dataspec_test_targets_file(spec):
    return spec['test_targets_file']

def dataspec_n_train(spec):
    return spec['n_train']

def dataspec_n_test(spec):
    return spec['n_test']

def dataspec_input_shape(spec):
    return spec['input_shape']

def dataspec_input_size(spec):
    return spec['input_size']

def dataspec_target_shape(spec):
    return spec['target_shape']

def dataspec_target_size(spec):
    return spec['target_size']

def dataspec_input_file_header_bufsize(spec):
    return spec['input_file_header_bufsize']

def dataspec_target_file_header_bufsize(spec):
    return spec['target_file_header_bufsize']

def dataspec_origin(spec):
    return spec['origin']

def dataspec_extra(spec):
    return spec['extra']

def dataspec_is_url(spec):
    return dataspec_origin(spec) == 'url'

def dataspec_url(spec):
    assert dataspec_is_url(spec), "dataspec must be a url spec"
    return dataspec_extra(spec)['url']

# %%

# DataSpec representation invariant
def check_dataspec(spec: DataSpec):
    raise NotImplementedError()

# %%

# DataSpec operations

def dataspec_fetch(spec, dest):
    assert dataspec_is_url(spec), "dataspec must be a url spec"
    return []

file_url = dataspec_url(MNIST_SPEC)
file_path = dataspec_train_inputs_file(MNIST_SPEC)

if os.path.exists(file_path):
    os.unlink(file_path)
url = os.path.join(file_url, file_path)
with urllib.request.urlopen(url) as req:
    with open(file_path, 'xb') as f:
        f.write(req.read())

# %%
# sample_shape = dataspec_input_shape(MNIST_SPEC)
# sample_size = dataspec_input_size(MNIST_SPEC)
# header_bufsize = dataspec_input_file_header_bufsize(MNIST_SPEC)
sample_shape = dataspec_target_shape(MNIST_SPEC)
sample_size = dataspec_target_size(MNIST_SPEC)
header_bufsize = dataspec_target_file_header_bufsize(MNIST_SPEC)
n_samples = 5

with gzip.open(file_path) as f:
    f.read(header_bufsize)  # skip header
    buf = f.read(sample_size * n_samples)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(n_samples, *sample_shape)

    if len(sample_shape) == 2:
        image = np.asarray(data[0])
        plt.imshow(image, cmap='gray')
        plt.show()

# dataspec_fetch(MNIST_SPEC, 'data/')

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
    parser = ArgumentParser(
        prog="xyn",
        description="xyn generates xynthetic data",
        epilog=unlines(LOGO),
        formatter_class=RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(required=True)

    parser_regression = subparsers.add_parser("reg")
    parser_regression.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    parser_regression.add_argument(
        "-n",
        "--noise",
        type=float,
        default=1e-2,
        help="Amount of gaussian noise used to generate data",
    )
    parser_regression.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                        generate the dataset",
    )
    parser_regression.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    parser_regression.add_argument(
        "n_features",
        type=int,
        help="Number of input features \
                        to generate data",
    )
    parser_regression.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs = '?',
        help="Number of output variables for each data sample"
    )
    parser_regression.set_defaults(func=make_regression_dataset)

    parser_sum = subparsers.add_parser("sum")
    parser_sum.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    parser_sum.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0,
        help="Amount of gaussian noise used to generate data",
    )
    parser_sum.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                        generate the dataset",
    )
    parser_sum.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    parser_sum.add_argument(
        "n_features",
        type=int,
        help="Number of input features \
                        to generate data",
    )
    parser_sum.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs='?',
        help="Number of output variables for each data sample",
    )
    parser_sum.set_defaults(func=make_sum_dataset)

    parser_mixture = subparsers.add_parser("mix")
    parser_mixture.add_argument(
        "-s",
        "--seed",
        type=int,
        default=546,
        help="Use the given random seed to generate data",
    )
    parser_mixture.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0,
        help="Amount of gaussian noise used to generate data",
    )
    parser_mixture.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all the intermediate data structures used to \
                                generate the dataset",
    )
    parser_mixture.add_argument(
        "n_samples", type=int, help="Number of total data samples to generate"
    )
    parser_mixture.add_argument(
        "n_features",
        type=int,
        default=2,
        nargs='?',
        help="Number of input features \
                        to generate data",
    )
    parser_mixture.add_argument(
        "n_outputs",
        type=int,
        default=1,
        nargs='?',
        help="Number of output variables for each data sample",
    )
    parser_mixture.set_defaults(func=make_mixture_dataset)

    args = parser.parse_args()

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
# - [ ] enhance: refactor redundant parser arguments for mixture dataset
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
# - [ ] enhance: refactor parser_regression, parser_sum repeated code
