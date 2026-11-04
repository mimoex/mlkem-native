# ML-KEM-768 SME2研究実装と再現手順

本ブランチは、同一鍵に対するML-KEM-768の一括DecapsをApple M4で評価する研究用実装です。
ベースはmlkem-nativeの **`0db30973c542e987a9499503a99c69aac7ab8c97`** です。
通常のML-KEM APIを維持し、SME2バッチAPIを独立して追加しています。

## 方式と評価範囲

提案方式は3段の打切りNTTで256係数を8個の32係数残存多項式へ変換し、
公開側のToeplitz行列を事前計算します。SME2のSMOPAで16要求を同時に処理し、
ZAを保持する処理とrolling workspaceを用います。NEONは要求側の前後変換、
剰余還元、逆変換に使います。逆変換後の出力転置は通常経路・rolling経路ともに、
ZAへの16-bit行ロードと列ストアで行います。通常のPKE復号などは上流AArch64実装を使います。

比較用Prepared NTTは `test/sme2_batch/prepared_ntt.{h,c}` にあります。
上流AArch64 NEONのNTT・積和・逆NTTをそのまま呼び出す研究用APIです。

| 比較方式 | 準備API / バッチAPI | 事前計算とscratch |
|---|---|---|
| 従来のfairness比較 | `prepare_ntt` / `ntt_fair_batch` | 公開行列・公開鍵のNTT表現を共有。要求側 `sp_ntt` と `cache` をバッチ内で再利用し、最後に消去 |
| O3最終報告の比較 | `prepare_ntt_public` / `ntt_public_batch` | 上記に公開行列3行と公開鍵のmulcacheを追加。要求側NTT後の既存の還元を保ち、公開側cacheを使用 |

後者は2026年10月16日のO3最終報告で使用した公開鍵mulcache方式です。
元の `ntt_public_cache.h` の処理を同じPrepared NTTファイルへ抽出しました。
従来の `bench_ntt_fairness.c` の方式とは区別してください。
両方式とも同一鍵の準備結果を複数要求・複数バッチで共有できます。
これらの関数は有効なポインタ、アラインメント、領域長、非重複、バッチ長を
呼出側で保証する内部比較用helperであり、新たな本番バックエンドではありません。

## 検証した環境

- Apple M4、macOS 26.6.2 / Darwin 25.6.0、arm64。
- Apple clang version 21.0.0 (`clang-2100.3.34.2`)。
- Streaming Vector Length (SVL) = 512 bits。各SME2呼出しで検査します。
- Cコードは `-O3 -mcpu=apple-m4 -std=c99 -Wall -Wextra -Werror`、LTOなし。
- バッチ比較ではAArch64算術を有効化し、FIPS202は共通のportable C構成です。
- Python 3.10以上。通常のテスト・ベンチマークは標準ライブラリのみを使います。

Apple M4でSME2を実行できるmacOSと、SME属性を解釈できるClangが必要です。
上記以外のCPU/OSでSME2が動作することは検証していません。
非SME2向けAArch64 NEON/portable C構成は、M4上でビルド・実行しています。

以下はリポジトリのルートで実行します。結果の既定出力先はGitの除外対象
`test/build/` です。スクリプトの `--output` は存在しないディレクトリを指定します。

## SME2・Prepared NTTのビルドと正確性

```sh
python3 test/sme2_batch/run_tests.py
python3 test/sme2_batch/build_matrix.py --output test/build/sme2-matrix
```

最初のコマンドは単一翻訳単位(SCU)と個別オブジェクトの両構成をビルド・リンクし、
SME2、従来Prepared NTT、公開鍵mulcache版のテストを実行します。
16〜256件の全16サイズで、正常・改変・混在暗号文を通常のML-KEM-768と比較します。
引数拒否、鍵の破損、使用領域の消去、未使用領域とガード値も検査します。
ZA出力転置は16-bit全域の値と出力境界を検査し、通常／rolling算術経路を
全16サイズで独立したschoolbook多項式積と比較します。
ビルド行列は512/768/1024、SME2/NEON/portable、複数翻訳単位、
SCUの全6パラメータ順序、context指定、Decaps無効化を含む20構成です。

従来Prepared NTT単独をSME2なしでビルドする例です。

```sh
mkdir -p test/build/prepared
clang -O3 -march=armv8-a -std=c99 -Wall -Wextra -Werror \
  -Imlkem -Imlkem/src -DMLK_CONFIG_PARAMETER_SET=768 \
  -DMLK_CONFIG_USE_NATIVE_BACKEND_ARITH \
  '-DMLK_CONFIG_ARITH_BACKEND_FILE="native/aarch64/meta.h"' \
  -DMLK_CONFIG_NO_RANDOMIZED_API \
  test/sme2_batch/test_prepared_ntt.c test/sme2_batch/prepared_ntt.c \
  mlkem/mlkem_native.c mlkem/mlkem_native_asm.S \
  -o test/build/prepared/test_prepared_ntt
test/build/prepared/test_prepared_ntt
```

M4向け比較と同じ条件では `-march=armv8-a` を `-mcpu=apple-m4` に置換します。
SME2 APIは `mlkem/src/native/aarch64/sme2/mlkem_sme2_decaps.h` の
`mlk_sme2_decaps_prepare`、`mlk_sme2_decaps_batch`、鍵/workspaceのclear関数です。
APIには名前空間と任意context引数の設定が適用されます。

## 上流APIの回帰テスト

```sh
CFLAGS='-mcpu=apple-m4' make -j4 CC=clang OPT=1 \
  BUILD_DIR=test/build/upstream-m4 run_func run_unit run_kat
CFLAGS='-march=armv8-a' make -j4 CC=clang OPT=1 \
  BUILD_DIR=test/build/upstream-neon run_func run_unit run_kat
CFLAGS='-march=armv8-a' make -j4 CC=clang OPT=0 \
  BUILD_DIR=test/build/upstream-portable run_func run_unit run_kat
```

各コマンドはML-KEM-512/768/1024を検査します。上流のテストは削除していません。

## O3最終方式による一括Decaps比較

```sh
# 正確性・安全性検査のみ
python3 test/sme2_batch/run_public_cache.py --verify-only
# 全16サイズ、31 ABBAブロックの簡易測定
python3 test/sme2_batch/run_public_cache.py --mode quick
# 全16サイズ、301ブロックの測定
python3 test/sme2_batch/run_public_cache.py --mode final
```

この入口は元の検査を維持しています。通常の差分テストに加え、cache破損の
negative control、ASan/UBSan、SVL不一致注入、SME ABIレジスタ保持、逆変換範囲、
5方式と独立schoolbook oracleの検査が成功してから測定します。
ASan/UBSanはネイティブアセンブリ内部を計装しません。

SME2とNTTは同じソース状態・同じバイナリ・同じ決定的入力で実行します。
公開値/鍵の準備はバッチ計測外です。FOの前後演算と使用領域の消去は計測内です。
`mach_continuous_time` によりns/reqを計測し、1観測8バッチ、warmup 2回とします。
各ブロックでバッチサイズをshuffleし、ABBA/BAAB順の各方式2観測を平均して対にします。
quickは5,000回、finalは20,000回のブートストラップでペア差中央値の95% CIを計算します。
速度比は各ペアの `NTT / SME2` の中央値です。1を超えるとSME2が高速です。
CIと3つの時系列区間を併記し、外れ値は削除しません。
鍵準備時間は別の `preparation.csv` に記録します。

過去のO3最終報告は算術・Decaps・準備の複数条件を共通ブロック内で混ぜた評価でした。
ここでは指定の一括Decaps用スクリプトを独立して再現します。
比較方式・演算・バッチ計時範囲は保持しますが、過去と同一の実行順序や温度条件、
過去の全論文表を再現するものではありません。

従来fairness方式とスケジュール比較も残しています。

```sh
python3 test/sme2_batch/run_fairness.py --mode quick
python3 test/sme2_batch/run_ntt_schedule.py --mode quick --batches 16,256
# 従来方式を全16サイズで測る例
python3 test/sme2_batch/run_fairness.py --mode final \
  --batches 16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256
```

指定なしの場合、従来fairness/scheduleのquickは80,128,160,192,224,256件、
finalは80〜256件を16件刻みで測ります。元のNEON16追加比較も、既存の正確性検査を
維持するため残しています。これは最終公開鍵mulcache方式とは別の補助比較です。

## CSVと来歴

- `*_raw.csv`: `comparison,batch,block,visit,slot,variant,ns_per_req`。
- `*_summary.csv`: 各方式の時間中央値、ペア差中央値、`ci_low,ci_high`、
  `paired_speedup_median`、勝数、サンプル数、3区間の中央値、判定。
- 公開鍵mulcache版は `measurement/` にCSV、`manifest.json` にソースhash、
  コマンド、コンパイラhash、依存ヘッダ、条件、検証結果を記録します。
  `git-head.log` が測定時コミット、`git-status-before.log` が未コミット差分の状態です。
- 従来fairness/scheduleは `metadata.json` にコミット、ソースhash、コンパイラ、
  ビルド行、バイナリhash、入力seed、ABBA seed、workspaceと計測条件を記録します。

quick測定は再現動作の確認用です。CIは同一セッション内の不確実性であり、
セッション間の温度・負荷・周波数変動を含む保証ではありません。
測定中は他のビルドやベンチマークを同時に実行しないでください。

## 5方式の算術比較

[5方式の手順](test/sme2_5way/README.md)を参照してください。
A: Direct SME2、B: 3段NTT + scalar、C: 3段NTT + SME2、
D: Native AArch64 NTT、E: Portable C NTTを比較します。
Decaps全体とは計時範囲が異なり、FO処理や明示zeroizeを含みません。

```sh
python3 test/sme2_5way/run.py --mode check
python3 test/sme2_5way/run.py --mode quick --batches 16,80,256
```

## 制約・upstreamとの差分

- SME2 APIはML-KEM-768、同一鍵、16の倍数で最大256要求、SVL=512専用です。
  CPUの実行時dispatchはありません。
- 鍵の共有は可能ですが、workspace/outputは呼出しごとに排他的に所有してください。
  ポインタの長さ・重なり・アラインメントは呼出側の契約です。
- 元の実装の秘密バッファ消去を保持します。全レジスタ、コンパイラspill、
  非アクティブなスタック領域、マイクロアーキテクチャ状態の完全消去を意味しません。
- SME2と比較helperの入口ガード費用は完全対称ではありません。
  SME2の引数/SVL検査は計測内です。比較helperはvalid-input前提です。
- 準備処理には元の診断出力も残ります。バッチ性能測定の外で実行します。
- 新規研究部分の形式証明・全プラットフォームの検証を主張するものではありません。

既存ファイルの変更は次の5ファイルです。

1. `mlkem/mlkem_native.c`: SME2 Cソースの取り込みとマクロの後処理。
2. `mlkem/mlkem_native_asm.S`: SME2アセンブリの取り込みとマクロの後処理。
3. `mlkem/src/sys.h`: コンパイル対象のSME2検出。
4. `scripts/autogen`: 決定的なソース列挙と複数パラメータセット用のマクロ後処理。
5. `test/mk/components.mk`: SME2個別オブジェクトのビルド対象追加。

通常のKEM、上流NEON演算、FIPS202、定数は変更していません。
自動生成C/アセンブリは更新済みの `scripts/autogen` から生成しています。
完全な上流自動生成には上流開発環境（pyparsing、PyYAML、rich、clang-format等）が必要です。

バックアップ、ビルド済みバイナリ、過去の測定CSV/ログ、ローカルDB、ZIP導入説明、
開発途中の実験ディレクトリは公開履歴に含めません。
