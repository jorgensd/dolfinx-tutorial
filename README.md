#Cheatsheet

Build book
```bash
pip3 install -U jupyter-book
jupyter-book build .
```

Push book
```bash
pip3 install ghp-import
ghp-import -n -p -f _build/html
```

# Requirements for dockerfile
https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html


# Build image for dockerhub
docker build . --tag dokken92/dolfinx_custom:tutorials

# Push to dockerhub
docker push dokken92/dolfinx_custom:tutorials