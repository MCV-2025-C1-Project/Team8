# Team8

Team8 project for C1 course of Master in Computer Vision (MCV)

## Dependencies

All the needed dependencies are listed in the _requirements.txt_ file.

Ideally, install them within a virtual environment. Create and activate it with:

```bash
python3 -m venv venv/
source myenv/bin/activate
# Windows: venv\Scripts\Activate
```

And then, from the root of the project, install the dependencies with:

```bash
pip install -r requirements.txt
```

Update the requirements.txt running:

```bash
pip freeze > requirements.txt
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
