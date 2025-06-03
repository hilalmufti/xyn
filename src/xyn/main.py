# %%
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial

import mlx.core as mx

while 'src' in os.getcwd():
    os.chdir('..')

# %%
MNIST_SPEC = {
    'train_inputs_file': 'train-images-idx3-ubyte.gz',
    'train_targets_file': 'train-labels-idx1-ubyte.gz',
    'test_inputs_file': 't10k-images-idx3-ubyte.gz',
    'test_targets_file': 't10k-labels-idx1-ubyte.gz',
    'n_train': 60_000,
    'n_test': 10_000,
    'origin': 'url',
    'extra': {
        'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    },
    'type': 'dataspec'
}

# %%
LOGO = [
    " _  _  _  _  __ _ ",
    "( \\/ )( \\/ )(  ( \\",
    " )  (  )  / /    /",
    "(_/\\_)(__/  \\_)__)",
]

# %%
def make_dataspec(train_inputs_file, train_targets_file, test_inputs_file,
                  test_targets_file, n_train, n_test, origin, extra):
    match origin:
        case 'url':
            assert 'url' in extra
            return {
                'train_inputs_file': train_inputs_file,
                'train_targets_file': train_targets_file,
                'test_inputs_file': test_inputs_file,
                'test_targets_file': test_targets_file,
                'n_train': n_train,
                'n_test': n_test,
                'origin': origin,
                'extra': extra,
                'type': 'dataspec'
            }
        case 'disk':
            raise NotImplementedError()
        case 'synthetic':
            raise NotImplementedError()

MNIST_SPEC = make_dataspec(
    train_inputs_file='train-images-idx3-ubyte.gz',
    train_targets_file='train-labels-idx1-ubyte.gz',
    test_inputs_file='t10k-images-idx3-ubyte.gz',
    test_targets_file='t10k-labels-idx1-ubyte.gz',
    n_train=60_000,
    n_test=10_000,
    origin='url',
    extra={
        'url': 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    }
)


# %%
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

def dataspec_origin(spec):
    return spec['origin']

def dataspec_extra(spec):
    return spec['extra']

def dataspec_is_url(spec):
    return dataspec_origin(spec) == 'url'

# %%
def dataspec_fetch(spec, dest):
    assert dataspec_is_url(spec), "dataspec must be a url spec"
    return []

dataspec_fetch(MNIST_SPEC, 'data/')

# %%
def unlines(xs: list[str]) -> str:
    return "\n".join(xs)

# %%
def unwords(xs: list[str]) -> int:
    return " ".join(xs)

# %%
def ndims(xs):
    match xs:
        case []:
            return 1
        case [x, *xs] if not isinstance(x, list):
            return 1
        case [x, *xs] if isinstance(x, list):
            return 1 + ndims(x)


# %%
def show_list_by(delim: str, xs: list):
    assert isinstance(xs, list)

    n = ndims(xs)
    if n == 1:
        return delim.join(map(str, xs))
    elif n == 2:
        return unlines(map(lambda x: show_list_by(delim, x), xs))
    else:
        raise Exception(f"Expected 1 or 2 dimensions, found {n}")


# %%
show_list = partial(show_list_by, " ")


# %%
def show_dict(mp):
    return unlines(k + "\n" + v for k, v in mp.items())


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
config = {
    'n_sample': 60_000,
    'n_features': 768,
    'n_outputs': 1,
    'noise_scale': 0,
    'weights': None,
}

seed = 546
key = mx.random.key(seed)

def make_mnist_dataset(config, key):
    ds = load_dataset('ylecun/mnist')['train']
    return ds[0]['image']

make_mnist_dataset(config, key)

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
        print(show_dict({k: show_list(v) for k, v in aux.items()}))
    print(show_header(range(1, args.n_features + 1), range(1, args.n_outputs + 1)))
    print(show_list(dataset.tolist()))


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
# - [ ] enhance: improve typechecking of ndims
# - [ ] enhance: improve function typechecking and precondition checking
# - [ ] enhance: implement unit and expect tests
# - [ ] enhance: implement property-based testing
# - [x] feat: multiple output variables
# - [ ] enhance: update readme for multiple output variables
# - [ ] feat: implement multi-dimensional classification dataset
# - [x] feat: implement splitting into train/test data (done by datasplit program)
# - [ ] enhance: refactor make_regression_dataset, make_sum_dataset
# - [ ] enhance: refactor parser_regression, parser_sum repeated code
