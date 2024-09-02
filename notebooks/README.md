# src ディレクトリ内のファイルへの path の通し方

下を各 notebook の先頭で読み込む

```
import sys
import configparser

# .ini ファイルをプロジェクトルートから読み込む
config = configparser.ConfigParser()
config.read('../config.ini')  # プロジェクトルートから実行する場合

# パスを取得
PROJECT_DIR = config['paths']['project_dir']
EXPT_RESULT_DIR = config['paths']['results_dir']

# PROJECT_DIR を Python のパスに追加
sys.path.append(PROJECT_DIR)
```