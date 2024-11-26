# Project Setup

**Prerequisites**: Make sure `uv` and `ollama` are installed on your system.

```bash
# create virtual environment
uv venv --python 3.12

# activate and install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt
```

Congratulations, you are ready to go! 🎉

# Managing Dependencies

### Adding a Dependency

```bash
# add the dependency to pyproject.toml
uv add <package-name>

# auto-generate a new requirements.txt file
uv pip compile pyproject.toml -o requirements.txt
```

### Removing a Dependency

```bash
# add the dependency to pyproject.toml
uv remove <package-name>

# auto-generate a new requirements.txt file
uv pip compile pyproject.toml -o requirements.txt
```

# Common Issues

> ResponseError: llama runner process has terminated: exit status 127

```bash
# Kill the ollama process
sudo systemctl stop ollama

# Explicity set the library path
export LD_LIBRARY_PATH=/usr/lib/ollama:$LD_LIBRARY_PATH

# Restart the ollama process
ollama serve
```
