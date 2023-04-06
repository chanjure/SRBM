pipreqs --force .
conda env export --no-builds | cut -f 1 -d '=' | grep -v "prefix:" > environment.yml
