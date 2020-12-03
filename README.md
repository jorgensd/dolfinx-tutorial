#Cheatsheet

Build book
```bash
pip3 install -U jupyter-book
jupyter-book build .
```

Push book
```bash
ghp-import -n -p -f _build/html
```