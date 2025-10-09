import nbformat
from pathlib import Path
nb = nbformat.read(Path('01ADS_Decision_Trees_2025.ipynb').open('r', encoding='utf-8'), as_version=4)
for idx, cell in enumerate(nb.cells):
    if 'algorithm=' in cell.get('source', ''):
        print(idx)
        print(cell['source'])
        print('---')
