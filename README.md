# Dataset preparation

Run from the project root with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m scripts.prepare_data
```

Calculate graph statistics from the prepared datasets:

```powershell
.venv/Scripts/python.exe -m scripts.analyze_graphs --offline
```
