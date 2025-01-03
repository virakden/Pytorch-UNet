#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  read -p "Kaggle username: " USERNAME
  echo
  read -p "Kaggle API key: " APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade

kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
unzip train_hq.zip
mv train_hq/* data/imgs/
rm -d train_hq
rm train_hq.zip

kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
unzip train_masks.zip
mv train_masks/* data/masks/
rm -d train_masks
rm train_masks.zip
