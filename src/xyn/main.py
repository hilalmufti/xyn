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
        return "\n".join(map(lambda x: show_list_by(delim, x), xs))
    else:
        raise Exception()

# %%
show_list = partial(show_list_by, " ")


# %%
def print_list(xs):
    print(show_list(xs))

# %%
def generic_print(x):
    if isinstance(x, list):
        print_list(x)
    else:
        print(x)

# %%
def print_dict(mp):
   for k, v in mp.items():
       print(k)
       generic_print(v)

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
    parser.add_argument("-s", "--seed",
                        type=int,
                        default=546,
                        help="Use the given random seed to generate data")
    parser.add_argument("-m", "--mean",
                        type=float,
                        default=1e-2,
                        help="Use the given mean to generate gaussian noise used \
                        to generate data")
    parser.add_argument("-a", "--all",
                        action='store_true',
                        help="Show all the intermediate data structures used to \
                        generate the dataset")
    parser.add_argument("n_samples", type=int, help="Number of total data samples to generate")
    parser.add_argument("n_features", type=int, help="Number of input features \
                        to generate data")
    parser.add_argument("n_outputs", type=int, help="Number of output variables for each data sample")

    args = parser.parse_args()

    seed = args.seed

    config = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_outputs": args.n_outputs,
        "noise_scale": args.mean
    }
    dataset, aux = make_regression_dataset(config, mx.random.key(seed))
    if args.all:
        print_dict(aux)
    print_list(dataset.tolist())

# TODO:
# - [x] feat: remove delim argument
# - [x] feat: add verbose mode
# - [x] feat: rename verbose to all
# - [ ] feat: add `xyn reg` subcommand
# - [x] feat: add sum dataset
# - [ ] fix: have sum dataset be exact sum (I think matmul is breaking it)
# - [ ] feat: add sum dataset for integral valued inputs
# - [ ] feat: add real-valued binary classification dataset
# - [ ] feat: add header display option
# - [ ] fix: fix scientific notation displaying
# - [x] fix: fix emacs/magit git setup
# - [x] feat: improve readme by showing how to use `xyn` with other unix commands
# - [ ] feat: Add examples to `-h` help
# - [ ] feat: improve typechecking of ndims
# - [ ] feat: typechecking
# - [ ] feat: tests
# - [x] feat: multiple output variables
# - [ ] fix: update readme for multiple output variables
# - [ ] feat: binary classification
# - [ ] feat: multi-dimensional classification
# - [ ] feat: train/test data
# - [ ] feat: specify where the true parameters come from
