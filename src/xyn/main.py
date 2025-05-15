# %%
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
import mlx.core as mx

# %%
LOGO = [
        " _  _  _  _  __ _ ",
        "( \\/ )( \\/ )(  ( \\",
        " )  (  )  / /    /",
        "(_/\\_)(__/  \\_)__)"
    ]

# %%
def unlines(xs):
    return "\n".join(xs)

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
        raise Exception()

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

    aux = {'weights': true_weights.tolist(),
           'noise': noise.tolist()}
    return mx.concatenate([inputs, targets], axis=1), aux

# %%
def make_sum_dataset(config, key):
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_outputs = config["n_outputs"]
    noise_scale = config["noise_scale"]

    assert n_outputs == 1, "sum dataset must have exactly one output"

    true_weights = mx.ones([n_features, n_outputs])

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples, n_outputs], key=subkey)

    targets = mx.matmul(inputs, true_weights) + noise

    aux = {'weights': true_weights.tolist(),
           'noise': noise.tolist()}
    return mx.concatenate([inputs, targets], axis=1), aux

# %%
def main() -> None:
    parser = ArgumentParser(prog="xyn",
                            description="xyn generates xynthetic data",
                            epilog="\n".join(LOGO),
                            formatter_class=RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(required=True)

    parser_regression = subparsers.add_parser('reg')
    parser_regression.add_argument("-s", "--seed",
                        type=int,
                        default=546,
                        help="Use the given random seed to generate data")
    parser_regression.add_argument("-m", "--mean",
                        type=float,
                        default=1e-2,
                        help="Use the given mean to generate gaussian noise used \
                        to generate data")
    parser_regression.add_argument("-a", "--all",
                        action='store_true',
                        help="Show all the intermediate data structures used to \
                        generate the dataset")
    parser_regression.add_argument("n_samples", type=int, help="Number of total data samples to generate")
    parser_regression.add_argument("n_features", type=int, help="Number of input features \
                        to generate data")
    parser_regression.add_argument("n_outputs", type=int, help="Number of output variables for each data sample")
    parser_regression.set_defaults(func=make_regression_dataset)


    parser_sum = subparsers.add_parser('sum')
    parser_sum.add_argument("-s", "--seed",
                        type=int,
                        default=546,
                        help="Use the given random seed to generate data")
    parser_sum.add_argument("-m", "--mean",
                        type=float,
                        default=1e-2,
                        help="Use the given mean to generate gaussian noise used \
                        to generate data")
    parser_sum.add_argument("-a", "--all",
                        action='store_true',
                        help="Show all the intermediate data structures used to \
                        generate the dataset")
    parser_sum.add_argument("n_samples", type=int, help="Number of total data samples to generate")
    parser_sum.add_argument("n_features", type=int, help="Number of input features \
                        to generate data")
    parser_sum.add_argument("n_outputs", type=int, help="Number of output variables for each data sample")
    parser_sum.set_defaults(func=make_sum_dataset)

    args = parser.parse_args()

    seed = args.seed

    config = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_outputs": args.n_outputs,
        "noise_scale": args.mean
    }
    dataset, aux = args.func(config, mx.random.key(seed))
    if args.all:
        print(show_dict({k: show_list(v) for k, v in aux.items()}))
    print(show_list(dataset.tolist()))

# TODO:
# - [x] enhancement: remove delim argument
# - [x] feat: implement verbose mode
# - [x] enhancement: rename verbose to all
# - [ ] enhancement: improve --help documentation for subcommands
# - [ ] enhancement: improve README documentation for subcommands
# - [ ] enhancement: remove n_outputs option for `xyn sum` subcommand
# - [ ] feat: implement `xyn reg` subcommand
# - [x] feat: implement sum dataset
# - [ ] fix: sum dataset be exact sum (I think matmul is breaking it), but is inexact
# - [x] feat: implement `xyn sum` subcommand
# - [ ] feat: implement sum dataset for integral valued inputs
# - [ ] feat: implement real-valued binary classification dataset
# - [ ] feat: implement option to display header
# - [ ] fix: array values should always print as decimals, but sometimes print in scientific notation
# - [x] fix: fix emacs/magit git setup
# - [x] feat: improve readme by showing how to use `xyn` with other unix commands
# - [ ] enhancement: Add examples to `-h` help
# - [ ] enhancement: improve typechecking of ndims
# - [ ] enhancement: improve function typechecking and precondition checking
# - [ ] enhancement: implement unit and expect tests
# - [ ] enhancement: implement property-based testing
# - [x] feat: multiple output variables
# - [ ] enhancement: update readme for multiple output variables
# - [ ] feat: implement multi-dimensional classification dataset
# - [x] feat: train/test data (done by datasplit program)
# - [ ] enhancement: refactor make_regression_dataset, make_sum_dataset
# - [ ] enhancement: refactor parser_regression, parser_sum repeated code
