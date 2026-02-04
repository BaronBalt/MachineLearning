# ML Component - First Model

Detta repo innehåller en första ML-komponent som tränar en Random Forest på Iris-datasetet.

## Struktur

- `data/` - rådata och processed data
- `src/` - återanvändbar kod
- `notebooks/` - workflow notebooks
- `models/` - versionerade modeller
- `registry/` - metadata om modeller
- `tests/` - unit tests
- `db` - databasfiler 

## 1️⃣ Setup på din dator

### 1.1 Ladda ned rätt Python version

Ladda ned [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) (äldre version för att den ska vara kompatibel med allt)

### 1.2 Klona repo

```bash
git clone https://github.com/BaronBalt/MachineLearning.git
cd MachineLearning
```

### 1.3 Skapa virtual environment

```bash
python -m venv MLenv
```

### 1.4 Aktivera environment

- Windows:
```powershell
MLenv\Scripts\Activate
```
- Mac/Linux (bash/zsh):
```bash
source MLenv/bin/activate
```

### 1.4 Installera dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
- `requirements.txt` innehåller alla paket som behövs: pandas, scikit-learn, torch, jupytext etc.

### 1.6 Starta JupyterLab

```bash
python -m jupyter lab
```

### 1.7 Testa att köra

Testa att köra `.ipynb` i `/notebooks` en efter en i ordningen 01, 02, osv och kolla på outputten som ges.
