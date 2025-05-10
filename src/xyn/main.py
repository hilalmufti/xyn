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
def show_by(delim: str, xs: list):
    assert isinstance(xs, list)

    n = ndims(xs)
    if n == 1:
        return delim.join(map(str, xs))
    elif n == 2:
        return "\n".join(map(lambda x: show_by(delim, x), xs))
    else:
        raise Exception()

# %%
show = partial(show_by, " ")

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
                        help="Use the given mean to generate gaussian noise used to generate data")
    parser.add_argument("-v", "--verbose",
                        action='store_true',
                        help="Enable verbose mode. More specifically, will also \
                        print out the exact true parameters and gaussian noise \
                        used to generate the data.")
    parser.add_argument("n_samples", type=int, help="Number of total data samples to generate")
    parser.add_argument("n_features", type=int, help="Number of input features to generate data")

    args = parser.parse_args()

    seed = args.seed
    noise_scale = args.mean
    n_samples = args.n_samples
    n_features = args.n_features
    is_verbose = args.verbose

    key = mx.random.key(seed)

    key, subkey = mx.random.split(key)
    true_params = mx.random.normal([n_features], key=subkey)

    key, subkey = mx.random.split(key)
    inputs = mx.random.normal([n_samples, n_features], key=subkey)

    key, subkey = mx.random.split(key)
    noise = noise_scale * mx.random.normal([n_samples], key=subkey)

    targets = mx.matmul(inputs, true_params) + noise

    lines = mx.concatenate([inputs, targets[:, None]], axis=1).tolist()

    if is_verbose:
        print("weights")
        print(show(true_params.tolist()))
        print("noise")
        print(show(noise.tolist()))
    print(show(lines))

# TODO:
# - [x] feat: remove delim argument
# - [x] feat: add verbose mode
# - [ ] fix: fix emacs/magit git setup
# - [ ] feat: improve readme by showing how to use `xyn` with other unix commands
# - [ ] feat: Add examples to `-h` help
# - [ ] feat: improve typechecking of ndims
# - [ ] feat: typechecking
# - [ ] feat: tests
# - [ ] feat: multiple output variables
# - [ ] feat: binary classification
# - [ ] feat: multi-dimensional classification
# - [ ] feat: train/test data
# - [ ] feat: specify where the true parameters come from
