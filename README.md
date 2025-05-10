# xyn

xyn (pronounced "syn") is a simple program to generate synthetic data.

```
 _  _  _  _  __ _
( \/ )( \/ )(  ( \
 )  (  )  / /    /
(_/\_)(__/  \_)__)
```

## Usage

Using `xyn` is really easy. To generate a synthetic dataset consisting of 5
samples with 2D inputs, simply run:
```
xyn 5 2
```
You should then see
```
1.4461036920547485 0.36945807933807373 -1.2877881526947021
1.1009156703948975 -0.02600148320198059 -1.0531200170516968
-0.17942558228969574 0.5433862209320068 0.31390732526779175
0.47421368956565857 1.405853033065796 -0.10289861261844635
-0.830163836479187 0.1463543325662613 0.819881796836853
```
The first two columns your input variables. The last column is your target
variable.

You can also customize the random seed, and amount of gaussian noise used to
generate the dataset. Checkout `xyn -h` for more.

## Install
Installation requires the Python programming language as well as the PyPI (`pip`) package manager.

### Step 1: Install Python and pip (skip this if it's already installed)
Python and PyPI are most likely already installed on your computer. Try executing:
```
python --version
pip --version
```
in your terminal to check if this is the case. 
If they're not installed (that is, one of the above commands yields an error), 
then I highly recommend you install them via `uv`. Run one of the following 
commands in the terminal to install `uv` on your computer:
```
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
More detailed instructions on installing `uv` can be found at `uv`'s [official
documentation](https://docs.astral.sh/uv/getting-started/installation/)

### Step 2: Install `xyn`
Now that `pip` is installed, we simply run one of the following in the terminal
to install `xyn`:
```
uv pip install https://github.com/hilalmufti/xyn.git # using uv
# or 
pip install https://github.com/hilalmufti/xyn.git # using PyPI
```

