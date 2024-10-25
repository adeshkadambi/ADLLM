# Setup Instructions

Assuming you installed ollmama version 0.4.0 or later, 
follow the instructions below to setup the environment.

1. Create virtual environment (python version 3.12.7):
```bash
python3 -m venv adl_venv
```

2. Activate virtual environment:
```bash
source adl_venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Common Issues


> ResponseError: llama runner process has terminated: exit status 127

```bash
# Kill the ollama process
sudo systemctl stop ollama

# Explicity set the library path
export LD_LIBRARY_PATH=/usr/lib/ollama:$LD_LIBRARY_PATH

# Restart the ollama process
ollama serve
```
