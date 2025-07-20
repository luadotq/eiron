# Extensible Intelligent Retrieval Open-source Navigator (EIRON)

<img width="2068" height="1847" alt="image_2025-07-20_06-43-38" src="https://github.com/user-attachments/assets/949a2708-95f0-422d-ade6-c9c79f00d9ba" />


**Multi-platform document search engine** for Windows and Linux/Unix, supporting both GUI and CLI modes.

---

## Features

- **Hybrid search**: Combines keyword and semantic analysis for high-quality results
- **Optimized memory management**: Efficient for both desktop and server environments
- **Automatic indexer selection**: Uses the best available indexer for your data
- **Full support for major document formats** (see table below)
- **Modern GUI** (PyQt5) and powerful CLI
- **Cross-platform**: All features available on both Windows and Linux/Unix
- **Fast indexing and search**: Handles large directories and millions of documents
- **Detailed file and index statistics**
- **Extensible architecture**: Easy to add new formats and search strategies

---

## Supported Formats

| Format  | Extension | Windows | Linux/Unix  | Notes                    |
|---------|-----------|---------|-------------|--------------------------|
| Text    | .txt      | Yes     | Yes         | Plain text files         |
| Markdown| .md       | Yes     | Yes         | Markdown documents       |
| Python  | .py       | Yes     | Yes         | Python source code       |
| JavaScript| .js      | Yes     | Yes         | JavaScript files         |
| HTML    | .html/.htm| Yes     | Yes         | Web pages                |
| CSS     | .css      | Yes     | Yes         | Stylesheets              |
| JSON    | .json     | Yes     | Yes         | Data files               |
| XML     | .xml      | Yes     | Yes         | XML documents            |
| PDF     | .pdf      | Yes     | Yes         | Full text extraction     |
| CSV     | .csv      | Yes     | Yes         | Spreadsheet data         |
| DOCX    | .docx     | Yes     | Yes         | Word documents           |
| XLSX    | .xlsx     | Yes     | Yes         | Excel spreadsheets       |

---

## Installation

### 1. Clone the repository
```shell
git clone https://github.com/luadotq/eiron.git
cd eiron
```

### 2. Install dependencies

> **Recommended:** Use a virtual environment (venv, conda, etc.)

```shell
pip install -r requirements.txt
```

---

## Usage

### GUI Mode

1. **Start the GUI:**
   ```shell
   python -m eirongui.py
   ```
   or
   ```shell
   python eirongui.py
   ```

2. **Index your documents:**
   - Go to the "Search" tab
   - Click "Select Folder" and choose the directory to index
   - (Optional) Adjust indexer settings in the "Settings" tab (choose indexer type, file, memory limits, etc.)
   - Click "Index" and wait for completion

3. **Search:**
   - Enter your query in the search box
   - Choose search mode (hybrid, keyword, semantic, exact)
   - Click "Search" to see results

4. **View statistics and logs:**
   - See index and memory stats in the "Settings" tab
   - Application logs are shown at the bottom of the window

### CLI Mode

1. **Index a directory:**
   ```shell
   python eironcli.py index --folder /path/to/documents --index-file my_index.bin --indexer optimized
   ```
   - `--indexer` can be `auto`, `standard`, or `optimized` (auto is recommended)

2. **Search:**
   ```shell
   python eironcli.py search --query "your search text" --index-file my_index.bin --indexer auto --mode hybrid
   ```
   - `--mode` can be `hybrid`, `keyword`, `semantic`, or `exact`

3. **Show index info:**
   ```shell
   python eironcli.py info --index-file my_index.bin
   ```

---

## Platform-Specific Features

- **Windows**: Office integration, higher memory usage for performance
- **Linux/Unix**: RLIMIT memory control, optimized for servers, lightweight mode
- **Both**: All major formats, hybrid/semantic search, GUI and CLI

---

## Indexer Types

- **Auto**: Automatically selects the best indexer based on your files
- **Standard**: Classic, simple indexer (compatible with all platforms)
- **Optimized**: Fast, memory-efficient, recommended for large datasets

---

## Tips & Best Practices

- For best performance, use the "Optimized" indexer (set in Settings tab or CLI)
- Always re-index after adding or removing many files
- Use the GUI for easy setup and monitoring, CLI for automation and scripting
- You can use multiple index files for different datasets
- Linux user's can get "Run of memory", just use RLIMIT control
---

## Development & Extensibility

- Modular codebase: add new file formats in `core/file_loader.py`
- Add new search strategies in `core/search.py` or `semantic/search.py`
- All tests are in the `tests/` folder and can be run directly

---

## Testing
Packaged with 8 auto-test scripts
Run any test directly:
```shell
python tests/test_context_search.py
```

---

## New Indexer alghorithm
New "optimized" indexer increases processing efficiency by up to 80%
<img width="1899" height="1837" alt="image_2025-07-20_06-44-30" src="https://github.com/user-attachments/assets/886738f8-55a0-4147-ad9f-64a943c60877" />

## License

This project is open-source and distributed under the GNU GPL 3 License
