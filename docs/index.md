# Welcome to IX-train

A website dedicated to training material for my ICRF.

## Bulding the docs

After pushing the changes to `main` GitHub branch, run the following command:
```text
mkdocs gh-deploy
```

## Environment installation

I recommend to setup the environment with all relevant packages using micromambda.

First clone the GitHub repo:
```text
git clone git@github.com:jonathon-langford/ix-train.git
cd ix-train
```

If you have not already installed micromambda then it can be done very simply with:
```text
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

You will be asekd a serious of questions to determine your preferred setup:
```text
Micromamba binary folder? [~/.local/bin] 
Init shell (bash)? [Y/n] Y
Configure conda-forge? [Y/n] Y
Configure conda-forge? [~/micromamba]
```

You may want to specify a certain location for the micromambda prefix (last line) as it can take up a reasonably large space. If this is successful you will receive some printout about appending lines to your `~/.bashrc`. Make sure to run:
```text
source ~/.bashrc
```

Test if `micromamba is available by running:
```text
micromamba --version
```

You can then setup the ix-train environment with:
```text
micromamba env create --prefix /vols/cms/jl2117/icrf/envs/ix-train -f environment.yaml
```

Replacing the prefix with where you want the environment binaries to be stored. Activate the environment with:
```text
micromamba activate /vols/cms/jl2117/icrf/envs/ix-train
```

