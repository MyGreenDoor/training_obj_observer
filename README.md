# Training Object Observer (Stereo) 入門ガイド

ステレオ画像から物体の姿勢や深度を推定するための PyTorch プロジェクト．エントリーポイントは `train_stereo_la.py` で，TOML 設定，TensorBoard，単一ノードの複数 GPU（torchrun）に対応している．

## 依存リポジトリ（重要）
- データローダは外部リポジトリ `la_loader` にある．クラウド環境だと無いことが多いので，学習やデータ読み込みはローカルで行う．
- 手元で `la_loader` をクローンし，パスを指定して `pip install -e /path/to/la_loader` しておく．

## 必要環境
- Python 3.10 以降（推奨）
- NVIDIA GPU＋CUDA（学習を高速化）．CPU でも動くが時間はかかる．

## セットアップ
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# 依存パッケージ
pip install -r requirements.txt
# データローダ（ローカルにクローン済みの la_loader を使う）
pip install -e /path/to/la_loader
```

## すぐ試す
```bash
# TensorBoard を起動（任意）
tensorboard --logdir outputs

# 単一 GPU で学習
python train_stereo_la.py --config configs/example_config.toml

# 複数 GPU（単一ノード）で学習
torchrun --nproc_per_node=4 --master_port=29501 train_stereo_la.py \
  --config configs/example_config.toml --launcher pytorch
```
- ログとチェックポイント: `outputs/run_debug/`
- TensorBoard ログ: `outputs/run_debug/tb/`
- 実際に使われた設定のコピー: `outputs/run_debug/config_used.toml`

## 設定ファイル（TOML）
`configs/example_config.toml` がサンプル．主なセクション：
- `[train]`: 出力先，エポック，バッチサイズ，学習率，AMP，勾配クリップなど
- `[data]`: 画像サイズ，ワーカー数，最大視差
- `[model]`: 特徴チャネル数，ポーズ予測の有無
- `[loss]`: 視差・ポーズ損失の重み

## フォルダ概要
- `train_stereo_la.py`: 学習メイン（設定読込 → モデル構築 → データローダ → 損失 → 最適化 → 学習ループ → ログ保存）
- `configs/`: TOML 設定
- `models/`: ネットワーク本体（例: `ssscflow2.py`）
- `losses/`: 損失関数（例: `loss_functions.py`）
- `utils/`: 分散処理や可視化のユーティリティ
- `outputs/`: ログやチェックポイントの出力先

## ログと可視化
- TensorBoard でスカラと画像（左右画像，推定視差，GT 視差）を確認できる．
- DDP で走らせると rank0 がチェックポイントを保存する．

## よくあるハマりどころ
- `ModuleNotFoundError: la_loader`: 外部リポジトリをローカルに用意し，`pip install -e` する．
- CUDA を掴まない: PyTorch を自分の CUDA バージョンに合わせて入れ直す．
- TensorBoard に何も出ない: `outputs/run_debug/tb` を指定しているか確認し，学習を少し進める．
