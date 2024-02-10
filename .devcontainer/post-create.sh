#!/bin/bash -eux

# if there are submodules
git submodule update --init --recursive

# install poetry and python packages
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
poetry config virtualenvs.create true
poetry install

# [optional] install starship prompt
curl -fsSL https://starship.rs/install.sh -o starship_install.sh
sh starship_install.sh -y
rm starship_install.sh

# [optional] configure fish shell
mkdir -p ~/.config/fish
echo "starship init fish | source" >> ~/.config/fish/config.fish
fish -c "set -U fish_greeting"
