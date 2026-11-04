# ML-KEM-768の5方式算術比較

Apple M4 / SVL=512、Clang、Python 3.10以上で実行します。
全方式・補助検証を `-O3 -mcpu=apple-m4` でビルドします。

| ID | 方式 | 内容 |
|---|---|---|
| A | Direct SME2 | 256×256の直接negacyclic畳み込み、16要求のSMOPA |
| B | 3段NTT + scalar | NEONの前変換、scalar Cの32係数Toeplitz積和と逆変換 |
| C | 3段NTT + SME2 | `mlk_sme2_arith_batch_rolling_preloaded` を直接利用 |
| D | Native AArch64 NTT | 上流NEONの完全NTT、cached積和、逆NTT |
| E | Portable C NTT | 上流portable Cを別namespaceでビルドし、同じNTT bridgeを利用 |

1要求につき3入力、4出力、多項式乗算12個の算術のみを比較します。
全方式とも係数表現の入出力を使い、公開値の準備は計時外です。
FO復号、SHAKE、再暗号化ノイズ生成、誤差加算、圧縮、暗号文比較、
Shared Secret導出、明示的zeroizeは含みません。
Decaps全体の評価には [README-SME2.md](../../README-SME2.md) の手順を使います。

```sh
python3 test/sme2_5way/run.py --mode check
python3 test/sme2_5way/run.py --mode quick --batches 16,80,256
python3 test/sme2_5way/run.py --mode final
```

既定の結果は `test/build/sme2_5way/{mode}-{timestamp}/` に生成されます。
`--output /path/to/new-directory` で変更できます。
`metadata.json`、正確なビルド行の `build.log`、検証と実行の `execution.log`、
観測値の `raw.csv`、集計の `summary.csv` を保存します。
quickは31ブロック、finalは301ブロックです。各ブロックで方式順をshuffleし、
その順と逆順で測ります。方式とバッチサイズごとに反復回数を較正します。

通常の実行でも、公開係数の中心化とZA overflowの回帰検査を先に実行し、
全方式の全係数をscalar Bと比較します。開始・中央・最後の要求は独立した
64-bit schoolbook oracleでも確認します。

全要求・全出力・全係数を独立oracleで検証してから簡易評価する例です。
これは動作確認用の短い1セッションであり、論文最終評価ではありません。

```sh
python3 test/sme2_5way/run_publication.py \
  --sessions 1 --samples 3 --batches 16,80,256 \
  --target-us 100 --bootstrap 1000 --cooldown 0
```

公開用評価はクリーンなコミットから次を実行します。

```sh
python3 test/sme2_5way/run_publication.py --expect-head "$(git rev-parse HEAD)"
```

既定は3セッション、301ブロック、全16サイズです。ソースとHEADの不変性、
コンパイラ、依存する研究ソースのhashを記録します。`--allow-dirty` は開発中の
検証に限ります。完全oracleや単体テストを省略するオプションは公開評価には使いません。
oracleは3つのseedと16/80/256件で、A/B/C/D/Eのすべてを検査します。

小さい独立検証を個別実行する例です。

```sh
mkdir -p test/build/sme2_5way-unit
for name in sanity_scalar test_partial test_direct_layout; do
  clang -O3 -mcpu=apple-m4 -std=c99 -Wall -Wextra -Werror \
    "test/sme2_5way/$name.c" -o "test/build/sme2_5way-unit/$name"
  "test/build/sme2_5way-unit/$name"
done
```

`summary.csv` は各方式のns/req中央値、Dとのペア差、95%ブートストラップCI、
3区間の中央値、Dに対する速度比を記録します。
算術のみの速度比をDecaps全体の速度比として解釈しないでください。
A/Bは研究比較用の実装で、各方式の最速実装を主張しません。
公開値の準備メモリも方式ごとに異なります。
