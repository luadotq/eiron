# Extensible Intelligent Retrieval Open-source Navigator (EIRON)
![image](https://github.com/user-attachments/assets/0d0c20cc-7b49-410c-86a6-8a9e2439c423)

**Multi-platform document search engine** with Windows and Linux/Unix optimized versions
## Platform-Specific Features

### Windows Version
- **Hybrid search** (keywords + semantic analysis)
- **COM object integration** for Office documents
- **Full Microsoft Office format support**
- **Higher memory utilization** for better performance

### Linux/Unix Version
- **Lightweight keyword search**
- **Word integration** for DOC files
- **RLIMIT memory control**
- **Optimized for server environments**

## Supported Formats Matrix

| Format  | Windows | Linux/Unix  | Notes                    |
|---------|---------|-------------|--------------------------|
| PDF     | Yes     | Yes         | Full text extraction     |
| DOCX    | Yes     | Yes         | Native support           |
| DOC     | Yes       | Yes         |                          |
| XLSX    | Yes       | Yes          | Sheet content            |
| CSV     | Yes       | Yes          | Fast processing          |
| HTML    | Yes       | Yes          | Clean text extraction    |
| PPTX    | Yes       | No          | Windows only             |
| RTF     | Yes      | Yes          |                          |
| TXT     | Yes       | Yes          |                          |

## Installation Guides

# Windows Installation

## 1. Clone repository
```shell
git clone https://github.com/luadotq/eiron.git
cd eiron/windows
```
## 2. Install dependencies
```shell
pip install -r requirements.txt
```
# Linux Installation

## 1. Clone repository
```shell
git clone https://github.com/luadotq/eiron.git
cd eiron/unix
```
## 2. Install Python packages
```shell
pip install -r requirements.txt
```
### How to use
1. Run eiron_win.py or eiron_unix.py (Depending on the operating system)
2. For first run please index directory
3. After indexing you can search
