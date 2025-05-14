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
def make_regression_dataset(n_samples, n_features, noise_scale, key):
    key, subkey = mx.random.split(key)
    true_weights = mx.random.normal([n_features], key=subkey)

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples], key=subkey)

    targets = mx.matmul(inputs, true_weights) + noise

    return mx.concatenate([inputs, targets[:, None]], axis=1)


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
                        help="Show all the data structures used to generate the \
                        dataset. More specifically, will also \
                        print out the exact true parameters and gaussian noise \
                        used for generation.")
    parser.add_argument("n_samples", type=int, help="Number of total data samples to generate")
    parser.add_argument("n_features", type=int, help="Number of input features \
                        to generate data")
    parser.add_argument("n_outputs", type=int, help="Number of output variables for each data sample")

    args = parser.parse_args()

    seed = args.seed
    noise_scale = args.mean
    n_samples = args.n_samples
    n_features = args.n_features
    n_outputs = args.n_outputs
    is_show_all = args.all

    key = mx.random.key(seed)

    key, subkey = mx.random.split(key)
    true_weights = mx.random.normal([n_features, n_outputs], key=subkey)

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples, n_outputs], key=subkey)

    targets = mx.matmul(inputs, true_weights) + noise

    dataset = mx.concatenate([inputs, targets], axis=1)
    if is_show_all:
        print("weights")
        print_list(true_weights.tolist())
        print("noise")
        print_list(noise.tolist())
    print_list(dataset.tolist())

# TODO:
# - [x] feat: remove delim argument
# - [x] feat: add verbose mode
# - [x] feat: rename verbose to all
# - [ ] feat: add `xyn reg` subcommand
# - [ ] feat: add sum dataset
# - [ ] feat: add real-valued binary classification dataset
# - [ ] feat: add header display option
# - [ ] fix: fix scientific notation displaying
# - [ ] fix: fix emacs/magit git setup
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
