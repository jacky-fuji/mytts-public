---
title: "日本語ボイスクローン6モデル比較: Irodori-TTS/Qwen3-TTS/Fish Speech等を音響分析"
emoji: "🎙️"
type: "tech"
topics: ["tts", "音声合成", "ai", "生成ai", "ローカルai"]
published: true
---

![ローカル日本語TTS / Voice Clone検証のビジュアルイメージ](/images/local-japanese-tts-voice-clone/tts_voice_clone_hero_visual_v2.png)

## このレポートの位置づけ

本稿は、2026年6月時点のローカル環境と公開モデルを前提に、個人が自分自身の声だけを対象に行った検証記録である。査読済み論文、製品評価レポート、第三者機関によるベンチマーク、専門家の鑑定ではない。

音響指標やグラフは比較を補助するためのものであり、本人性、自然さ、品質、各モデルの一般性能を保証するものではない。結果は、録音環境、入力文、参照音声、モデルのバージョン、推論設定、PC構成に強く依存する。したがって、ここでの評価は「この条件で試したらこう聞こえ、こう測定された」という実験メモとして読むのが適切である。

また、音声クローンは本人の同意がある声だけを扱うべきであり、本稿は第三者の声真似、なりすまし、誤認を誘う用途を推奨しない。

## 概要

:::message
先に生成音声を聞き比べたい場合は、GitHub Pagesの[試聴ページ](https://jacky-fuji.github.io/mytts-public/)で、同一文のモデル比較とIrodori-TTSの長文サンプルを確認できます。補足資料と公開可能なスクリプトは[GitHubリポジトリ](https://github.com/jacky-fuji/mytts-public)にまとめています。この記事本文では、録音、モデル比較、音響分析、得られた知見を順に整理します。
:::

ローカルPCだけで、日本語のText-to-SpeechとVoice Cloneを試した。目的は、本人の参照音声を使って、自然な日本語音声をどこまで生成できるかを確認することだった。

クラウドAPIではなくローカル実行にこだわった理由は、試行回数を増やしやすいこと、音声データを手元に置けること、GPU環境での現実的な速度と品質を把握したかったことにある。音声クローンは本人の声だけを対象にしており、第三者の声真似やなりすまし用途は対象外である。

結論は次のとおり。

| 観点 | 良かったモデル | コメント |
|---|---|---|
| 聴感上の自然さ | Fish Speech S2 Pro | 最も自然に聞こえた。ただし生成は非常に遅い |
| 速度と品質のバランス | Irodori-TTS 600M VoiceDesign | 日本語向け、軽量、長めの生成にも使いやすい |
| 内容保持と短文安定性 | VoxCPM2 ultimate | ASRとDTW系列類似度が高い。抑揚の違和感は残る |
| 短文の試行 | Qwen3-TTS 1.7B | 短文では良い。技術語・数字・長文は注意が必要 |
| 条件次第で候補 | CosyVoice2 | H4n Pro参照音声では改善したが、声質はトップ群に一歩及ばない |
| 本稿では不採用 | F5-TTS / GPT-SoVITS / Style-Bert-VITS2 | F5-TTSは日本語が崩れ、GPT-SoVITSはノイズ感が強かった。Style-Bert-VITS2は学習完了まで到達できなかった |

大きな学びは、モデルだけで決まるわけではないということだった。録音品質、参照音声の長さ、読み仮名への正規化、短文チャンク化、後処理、評価方法のほうが結果に強く効く場面が多い。

## 実行環境

検証環境は以下。

| 項目 | 内容 |
|---|---|
| OS | Microsoft Windows 11 Home 10.0.26200 / 64bit |
| マザーボード | ASRock X870 Taichi Creator |
| CPU | AMD Ryzen 7 7700 / 8 cores / 16 threads |
| RAM | 約64GB |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16,311 MiB |
| NVIDIA Driver | 596.36 |
| 録音機材 | Zoom H4n Pro |
| 録音形式 | WAV / 48kHz / 24bit / stereo |

GPU情報は次のように確認した。

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

CPUとOSはPowerShellで確認できる。

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption,Version,BuildNumber,OSArchitecture

Get-CimInstance Win32_Processor |
  Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
```

## リポジトリ構成

検証はGit管理し、生成音声、入力テキスト、比較結果、スクリプトを分けて保存した。WAVは大きくなりやすいため、公開・共有する場合は必要なサンプルだけを選ぶのが現実的である。

```text
mytts/
  docs/                  記事、比較レポート、図表
  docs/comparisons/      モデル比較の詳細レポート
  docs/assets/           スペクトログラムやF0グラフ
  samples/               読み上げ原稿、参照音声、manifest
  scripts/               生成、分割、正規化、ASR、音響分析
  outputs/               生成音声、文字起こし、分割ログ
  tools/                 各TTSモデルのローカル環境
```

初期化は通常のGitで十分。

```powershell
git init
git status
git add docs scripts samples
git commit -m "Initialize local TTS evaluation workspace"
```

## 録音データ

最初に重要だったのは、参照音声の品質だった。会議用スピーカーマイクでも試せるが、本人声に寄せる検証では録音品質がそのまま上限になる。そこで、Zoom H4n Proを使い、48kHz/24bit WAVで録音した。

H4n Proは96kHz録音も可能だが、本検証では48kHzで十分と判断した。多くのTTSモデルは内部で24kHzや32kHzに変換するため、96kHzに上げるより、部屋のノイズ、口とマイクの距離、入力レベル、読み直し管理を安定させるほうが効く。

読み上げ文は100文用意した。内容は日常文だけでなく、日本語TTSが苦手にしやすい要素を意図的に含めた。

- 数字、日付、割合
- 「七日」「九日」など読みが揺れやすい語
- 「生」のように文脈で読みが変わる語
- `GPU`、`CPU`、`API`、`CUDA` などの技術語
- URL、ファイル名、英数字
- 清音、濁音、半濁音、拗音
- 説明調、会話調、プレゼン調

:::details 追加録音用の読み上げテキスト100文を表示する
1. `001` おはようございます。今日は音声合成の品質を、短い文章で確認します。
1. `002` それでは、昨日の午後三時ごろに起きた問題から順番に見ていきます。
1. `003` この文章では、句読点、間、息継ぎ、そして語尾の自然さを確認します。
1. `004` なるほど、それなら一度、短い文章に分けて試したほうがよさそうですね。
1. `005` はい、では次に、数字と固有名詞を含む文章をもう一度確認しましょう。
1. `006` 今日は少し寒いですが、部屋の中は静かで、録音には向いています。
1. `007` さっきの出力は少し不自然でしたが、前回よりは改善していました。
1. `008` この声が自分らしく聞こえるかどうかを、あとで落ち着いて確認します。
1. `009` 焦らずに、同じ調子で、最後まで読み上げていきます。
1. `010` 一文ごとに短く区切ると、聞き比べがしやすくなります。
1. `011` 六月七日、九月九日、十一月二十日、十二月三十一日を読み上げます。
1. `012` 一日、二日、三日、四日、五日、六日、七日、八日、九日、十日です。
1. `013` 十四日、二十四日、二十日、三十日、三十一日も確認します。
1. `014` 一人、二人、三人、四人、五人、六人、七人、八人、九人、十人です。
1. `015` 一つ、二つ、三つ、四つ、五つ、六つ、七つ、八つ、九つ、十です。
1. `016` 一本、二本、三本、四本、五本、六本、七本、八本、九本、十本です。
1. `017` 一杯、二杯、三杯、四杯、五杯、六杯、七杯、八杯、九杯、十杯です。
1. `018` 午前一時十五分から、午後二時四十五分まで、会議を行います。
1. `019` 価格は税込み一万二千三百四十五円で、送料は六百六十円です。
1. `020` 成功率は九十九点八パーセントで、誤差は〇点二パーセントです。
1. `021` 在庫は残り三十六個で、次回入荷は七月中旬の予定です。
1. `022` 電話番号は、〇三、四五六七、八九〇一です。
1. `023` 部屋番号は二〇三号室で、受付は一階にあります。
1. `024` 一分、二分、三分、四分、五分、十分、十五分、三十分です。
1. `025` 一秒、二秒、三秒、四秒、五秒、十秒、十五秒、三十秒です。
1. `026` RTX 5060 Ti、VRAM 16GB、CUDA、Python、PyTorch を確認します。
1. `027` API サーバー、GPU メモリ、CPU 使用率、ログファイルを順番に見ます。
1. `028` URLは、エイチティーティーピーエス、コロン、スラッシュ、スラッシュ、example dot com です。
1. `029` GitHub、Hugging Face、ModelScope、ComfyUI、AivisSpeech を比較します。
1. `030` マイク、オーディオインターフェース、ボイスレコーダー、会議用スピーカーを使います。
1. `031` バージョンは、ゼロ点六ビー、いってんななビー、にてんぜろ、さんてんぜろです。
1. `032` ファイル名は、myvoice、ref seven seconds、round two、test sample です。
1. `033` エラーコードは四〇四、五〇〇、五〇三で、原因をログから確認します。
1. `034` この設定では、top p、temperature、max new tokens を変更します。
1. `035` 読み上げ結果は wav ファイルとして outputs フォルダに保存します。
1. `036` 母音は、あ、い、う、え、お。もう一度、あ、い、う、え、お。
1. `037` かきくけこ、さしすせそ、たちつてと、なにぬねの。
1. `038` はひふへほ、まみむめも、やゆよ、らりるれろ、わをん。
1. `039` がぎぐげご、ざじずぜぞ、だぢづでど、ばびぶべぼ、ぱぴぷぺぽ。
1. `040` きゃ、きゅ、きょ。しゃ、しゅ、しょ。ちゃ、ちゅ、ちょ。
1. `041` にゃ、にゅ、にょ。ひゃ、ひゅ、ひょ。みゃ、みゅ、みょ。
1. `042` りゃ、りゅ、りょ。ぎゃ、ぎゅ、ぎょ。じゃ、じゅ、じょ。
1. `043` きっと、さっき、チェック、ぴったり、ゆっくり、はっきり。
1. `044` こんにちは、問題、音声、確認、変更、生成、安定、反応。
1. `045` コーヒー、データ、ユーザー、サーバー、メーター、エラー、パラメーター。
1. `046` 静かな室内で、小さな機械が、細かな記録を次々に残します。
1. `047` 青い空、白い雲、静かな部屋、明るい声、低い声を順番に読みます。
1. `048` 赤い朝日が、青い海の向こうから、ゆっくり昇ってきました。
1. `049` 新しい車両は、急な坂道でも静かに走り、乗客を安全に運びます。
1. `050` ざわざわした会場で、発表者は落ち着いて、重要な説明を続けました。
1. `051` これは説明動画の冒頭で使う、落ち着いた読み上げの文章です。
1. `052` まず目的を確認し、次に手順を説明し、最後に結果を比較します。
1. `053` 録音データの品質が低いと、生成音声も不安定になることがあります。
1. `054` 一方で、短い検証では、完璧な環境よりも早く傾向をつかむことを優先します。
1. `055` 聞き手が内容を追いやすいように、読点では短く止まり、句点では少し長めに止まります。
1. `056` 長い文章では、途中で声が揺れたり、語尾が不自然になったりしないかを確認します。
1. `057` このモデルは声質をある程度まねできますが、発音やアクセントは入力文にも左右されます。
1. `058` 数字や英字の読みが崩れる場合は、読み仮名に近い形で入力します。
1. `059` 参照音声が短すぎる場合、声質は似ても、話し方が安定しないことがあります。
1. `060` 参照音声が長すぎる場合、必要な特徴以外の癖まで拾ってしまうことがあります。
1. `061` この声、ちゃんと自分らしく聞こえていますか。
1. `062` 少し硬いかもしれませんが、読み上げ用なので、普段よりもゆっくり話しています。
1. `063` もし長文が単調に聞こえるなら、別のモデルや別の録音条件を試してみます。
1. `064` うまくいけば、任意の日本語テキストを、自分の声に近い音声で保存できるようになります。
1. `065` これは、質問ですか。それとも、次の作業への指示でしょうか。
1. `066` はい、わかりました。もう一度、同じ文章を別の設定で生成して比較します。
1. `067` たしかに、その読み方だと少し不自然に聞こえるかもしれません。
1. `068` では、まず短い文章だけを出力して、声の近さを確認しましょう。
1. `069` そのあとで、数字や日付を含む文章を使って、発音の癖を見ます。
1. `070` 結果がよければ、同じ設定で少し長い文章も試してみます。
1. `071` これは普通の会話に近い、自然な声の高さで読んでいます。
1. `072` これは少しだけゆっくり、はっきり説明するように読んでいます。
1. `073` これは少しだけ早口ですが、言葉がつぶれないように注意しています。
1. `074` これは少し低めの声ですが、無理に作った声ではありません。
1. `075` これは少し明るめの声ですが、大げさな演技にはしません。
1. `076` これは少し疑問に思っている話し方です。本当にこの設定で大丈夫でしょうか。
1. `077` これは少し嬉しそうに話す文章です。思ったより自然に聞こえて安心しました。
1. `078` これは少し困った感じの文章です。原因が分からないので、もう一度確認します。
1. `079` これは落ち着いて報告する文章です。結果だけを簡潔に共有します。
1. `080` これは最後に確認する文章です。録音が終わったら、ファイル名を保存します。
1. `081` 今日、明日、あさって、昨日、おととい、来週、先週、再来週。
1. `082` 今年、去年、一昨年、来年、再来年、上半期、下半期、年度末。
1. `083` 東京の日本橋と大阪の日本橋では、読み方が変わる場合があります。
1. `084` 人気のある商品と、人の気配がない場所では、同じ漢字でも読み方が違います。
1. `085` 上手な説明と、上手の席では、文脈によって読み方が変わります。
1. `086` 十分な時間があります。十分後にもう一度確認します。
1. `087` 生き物の声と、生ビールの注文では、生の読み方が変わります。
1. `088` 今日は雨ですが、明日は晴れる見込みです。気温は二十八度まで上がります。
1. `089` この地域では、朝晩の冷え込みが強く、日中との温度差が大きくなります。
1. `090` 次の電車は、八時五分発、快速、東京行きです。
1. `091` 小さな会社でも、大きな仕組みを作れば、作業の手間を減らせます。
1. `092` 古い設定を残したまま、新しい設定を追加すると、原因の切り分けが難しくなります。
1. `093` まず一つだけ変更し、結果を確認してから、次の変更に進みます。
1. `094` 録音した音声は、短い区間に分割し、本文と完全に一致させます。
1. `095` 噛んだ場所や言い直した場所は、学習用データから外したほうが安全です。
1. `096` ノイズが入った場合は、その文だけもう一度読み直します。
1. `097` 同じマイク、同じ距離、同じ音量で録ると、比較がしやすくなります。
1. `098` この録音は、声の近さ、日本語の自然さ、数字の読み、長文の安定性を見るためのものです。
1. `099` ここまで読めば、次の実験に使うための十分なサンプルがそろいます。
1. `100` 以上で、追加録音用の読み上げテキストを終了します。
:::


録音は100ファイルに分けず、25文前後ずつ4本の長いWAVとして録った。その後、VADとASRを使って100個のチャンクに切り出した。

ここでいうVADはVoice Activity Detection、つまり音声区間検出である。長い録音の中から、声が入っている区間と無音区間を機械的に分けるために使った。ASRはAutomatic Speech Recognition、自動音声認識のことで、切り出したチャンクが想定した読み上げ文と対応しているかを文字起こしで確認するために使った。

| ソース | 対象文 | 長さ | Peak | RMS | 候補数 | 採用数 | 要確認 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MONO-000.wav` | 001-025 | 232.975s | -3.73 dBFS | -32.52 dBFS | 31 | 25 | 1 |
| `MONO-001.wav` | 026-050 | 267.008s | -0.00 dBFS | -32.19 dBFS | 31 | 25 | 13 |
| `MONO-002.wav` | 051-076 | 210.073s | -7.32 dBFS | -35.43 dBFS | 30 | 26 | 0 |
| `MONO-003.wav` | 077-100 | 207.755s | -15.48 dBFS | -37.74 dBFS | 32 | 24 | 1 |

切り出しは次のように実行した。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\slice_h4n_round2_recording.py `
  --input samples\voice\MONO-003.wav `
  --start-id 77 `
  --end-id 100 `
  --use-itn
```

参照音声も、切り出したH4n音声から作った。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\build_round2_reference_clips.py `
  --plan-set h4n `
  --input-dir samples\voice\h4n_round2_wav `
  --input-prefix h4n_round2_
```

よく使った参照音声は以下。

| 参照音声 | 長さ | 主な用途 |
|---|---:|---|
| `h4n_ref10s_neutral_091_093.wav` | 9.460s | Fish Speech |
| `h4n_ref20s_neutral_091_094.wav` | 21.030s | Irodori-TTS / Qwen3-TTS / CosyVoice2 / VoxCPM2 |
| `h4n_ref38s_japanese_edge_081_087.wav` | 37.730s | 発音・読みの確認 |

参照音声は長ければ長いほど良いわけではなかった。8秒から20秒程度の自然な音声が扱いやすい。長すぎる参照音声は、モデルによっては制約に引っかかったり、本文との対応付けが不安定になったりする。

## 試したモデル

対象にしたモデルと位置づけは以下。

| モデル / ツール | 方式のイメージ | 評価 |
|---|---|---|
| AivisSpeech / AIVIS系 | ローカルTTSの入口 | GUIや既存音声の確認用途 |
| GPT-SoVITS | 参照音声 + 学習/推論 | ノイズ感が強く、本稿の用途では不採用 |
| F5-TTS | zero-shot voice cloning | 日本語が中国語寄りに崩れ、不採用 |
| CosyVoice2 | 参照音声条件付け | H4n参照では改善。候補には残る |
| VoxCPM2 | 参照音声条件付け | 内容保持が強い。抑揚は要確認 |
| Qwen3-TTS 1.7B | 参照音声条件付け | 短文では良い。長文・技術語で崩れやすい |
| Fish Speech S2 Pro | 高品質TTS/voice clone | 最も自然。生成は非常に遅い |
| Irodori-TTS 500M / 600M VoiceDesign | 日本語向けTTS、caption制御 | 速度と品質のバランスが良い |
| Style-Bert-VITS2 | 日本語TTS、学習型 | データセット作成までは可能。学習完了は未達 |

ここで言う「voice clone」は、本人の参照音声を使って声質や話し方を寄せるという意味である。すべてのモデルで専用モデルを学習したわけではない。中心は、学習済みモデルに参照音声を渡すzero-shot voice cloning、reference conditioning、VoiceDesignである。

## モデルの出典とライセンス

以下は2026年6月16日時点で、公式リポジトリ、公式サイト、Hugging Faceモデルカード、技術報告を確認した内容である。ライセンスは変更される可能性があるため、公開利用や商用利用の前には、リンク先の`LICENSE`とモデルカードを再確認する必要がある。

| モデル | 開発元・背景 | 公式情報 | ライセンスと利用メモ |
|---|---|---|---|
| AivisSpeech / AIVIS系 | Aivis Projectによる日本語向けローカル音声合成環境。今回の検証ではGUIや既存音声の確認用途として使った。 | [Aivis Project公式サイト](https://aivis-project.com/) | 公式サイトでは、AivisSpeech自体は個人・法人・商用を問わず基本的に自由に使えると説明されている。ただし、AivisHub上の音声モデルは`ACML`、`ACML-NC`、`CC0`、カスタムライセンスなどに分かれるため、モデルごとの条件確認が必要。 |
| GPT-SoVITS | RVC-Bossによるfew-shot voice cloning系TTS。短い音声データから音声合成を試せる。 | [GitHub](https://github.com/RVC-Boss/GPT-SoVITS) | リポジトリのコードはMITライセンス。研究用途や自分の声での検証は扱いやすい。一方で、事前学習済み重み、学習データ、第三者の声の権利は別問題なので、公開・商用利用では同意と配布条件を確認する必要がある。 |
| F5-TTS | SWividらによるflow matching / DiTベースのzero-shot TTS。英中を中心に広く試されている。 | [GitHub](https://github.com/SWivid/F5-TTS)、[Project page](https://swivid.github.io/F5-TTS/)、[paper](https://arxiv.org/abs/2410.06885) | コードはMITライセンスだが、公式READMEでは事前学習済みモデルはEmiliaデータセット由来のため`CC-BY-NC`とされる。研究・非商用検証は可能だが、この重みを使った商用利用には向かない。 |
| CosyVoice2 | FunAudioLLM Team / SpeechLab@Tongyi, Alibaba Groupによる多言語TTS。zero-shot、cross-lingual、streaming推論などを重視している。 | [GitHub](https://github.com/FunAudioLLM/CosyVoice)、[CosyVoice2 demo](https://funaudiollm.github.io/cosyvoice2/)、[Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)、[paper](https://arxiv.org/abs/2412.10117) | リポジトリとHugging FaceモデルカードはApache-2.0。研究用途・商用用途とも比較的扱いやすい。ただし、配布物に含まれるサンプル音声やデモ素材の扱いは、モデル利用とは別に確認するのが安全。 |
| VoxCPM2 | OpenBMBによる2B規模のTTS。MiniCPM系のバックボーンを使い、30言語、48kHz出力、音声クローン、Voice Designをうたう。 | [GitHub](https://github.com/OpenBMB/VoxCPM)、[Hugging Face](https://huggingface.co/openbmb/VoxCPM2)、[docs](https://voxcpm.readthedocs.io/)、[paper](https://arxiv.org/abs/2606.06928) | 公式READMEでは重みとコードがApache-2.0で、commercial-readyと説明されている。研究用途・商用用途とも使いやすい候補。 |
| Qwen3-TTS 1.7B | Qwen Team / Alibaba CloudによるQwen3系TTS。技術報告では、短い参照音声によるvoice cloning、記述ベース制御、多言語対応を掲げている。 | [Qwen公式サイト](https://qwen.ai/)、[Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) | 技術報告ではモデルとtokenizerをApache-2.0で公開すると説明されている。研究用途・商用用途とも扱いやすい方向だが、実際に使う配布元のモデルカードとComfyUIノード側の条件も確認する。 |
| Fish Speech S2 Pro | Fish Audio / 39 AI, Inc.による高品質TTS。S2 Proは多言語・大規模データ・高い自然性を売りにしている。 | [GitHub](https://github.com/fishaudio/fish-speech)、[Hugging Face](https://huggingface.co/fishaudio/s2-pro)、[Fish Audio](https://fish.audio/)、[paper](https://arxiv.org/abs/2603.08823) | Fish Audio Research License。研究・非商用利用は無料で許可されるが、商用利用にはFish Audioとの別ライセンスが必要。今回の品質は最も高かったが、公開プロダクト化ではライセンス確認が必須。 |
| Irodori-TTS 500M / 600M VoiceDesign | Aratako / Chihiro Arata氏による日本語向けTTS。v3系はzero-shot voice cloning、VoiceDesign、Speaker Inversion、LoRAなどの経路を持つ。 | [GitHub](https://github.com/Aratako/Irodori-TTS)、[500M v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)、[600M v3 VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign) | コードとv3系モデルカードはMITライセンス。研究用途・商用用途とも扱いやすい。一方で、モデルカードでは本人同意のない声真似、なりすまし、誤情報用途が禁止されている。漢字読みは弱めなので、日本語では読み仮名寄せが有効。 |
| Style-Bert-VITS2 | litagin02氏による日本語TTS。Bert-VITS2系を土台に、感情や話し方のスタイル制御を扱う。 | [GitHub](https://github.com/litagin02/Style-Bert-VITS2) | コードはAGPL-3.0。研究利用は可能だが、サービス化や改変配布ではAGPLのソース公開義務に注意が必要。デフォルトモデルや派生元モデルの利用規約も別途確認する。 |

TTSでは、コードのライセンス、モデル重みのライセンス、学習データの権利、生成音声の利用条件が一致しないことがある。特にvoice cloningでは、モデルがオープンでも、第三者の声を本人同意なしに再現することは避けるべきである。本検証は本人の録音音声のみを使っているため、声の権利面は比較的整理しやすい。

## 評価方法

TTSでは、評価軸を分けないと判断を誤る。文字通り読めているか、声が本人に近いか、自然に聞こえるか、長文で壊れないか、生成速度が現実的かは別問題である。

本検証では、次の4つを使った。

| 評価 | 見ているもの | 読み方 |
|---|---|---|
| 聴感評価 | 本人らしさ、自然さ、抑揚、違和感 | 最終判断では最重要。ただし主観的 |
| ASR一致率 | 生成音声を文字起こししたとき、期待テキストと合うか | 高いほど本文を保てている。ただし声質は測れない |
| 音響特徴量 | MFCC、メルスペクトログラム、F0、フォルマントなど | 元音声との物理的な近さを補助的に見る |
| 生成コスト | 生成時間、RTF、VRAM使用量 | 実運用できるかを見る |

ASRはcontent preservation、つまり「読ませた文がどれだけ保たれているか」を見るために使った。声質や本人らしさはASRでは分からない。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\transcribe_generated_outputs.py `
  --input-root outputs\similarity_eval `
  --sample-dir samples\text `
  --output-prefix similarity_eval_all_models_asr `
  --use-itn
```

音響分析は次のスクリプトで行った。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\analyze_acoustic_similarity_deep.py
```

## 音響指標の読み方

音響分析では、次の指標を使った。ここは専門用語が多いので、簡単に補足する。

| 指標 | 意味 | 良い値の読み方 |
|---|---|---|
| Composite | 複数の音響指標を重み付きでまとめた総合スコア | 0から1で、高いほど元音声に近い。ただし0.001程度の差で勝敗は決めない |
| Timbre | 音色の近さ。MFCC、スペクトル特徴、フォルマントを混ぜたもの | 高いほど声質の傾向が近い。ただし主観的な本人らしさそのものではない |
| Frame/DTW | 同じ文を読んだときの時系列特徴の近さ | 高いほど、時間伸縮後の音響特徴が近い |
| Prosody | 抑揚、ピッチの動きの近さ | 高いほどイントネーションが近い |
| Formants | 声道共鳴、特に母音の響きに関係する特徴 | 高いほど母音・声道の響きが近い |
| Duration ratio | 生成音声の長さ / 元音声の長さ | 1.0に近いほど話速や間が近い |
| Median F0 delta | 生成音声と元音声のピッチ中央値の差 | 0Hzに近いほど声の高さの中心が近い |
| F0 corr | F0輪郭の相関 | 高いほどピッチの上下パターンが近い |
| Formant relative distance | フォルマントの相対距離 | 低いほど声道共鳴が近い |

MFCCは、声のスペクトル包絡を低次元で表す特徴量である。ざっくり言うと、声質や声道の形に関係する情報を数値化する。

メルスペクトログラムは、人間の聴覚に近い周波数軸で音のエネルギーを見たものだ。図にすると、どの周波数帯がいつ強いかが分かる。

F0は基本周波数で、声の高さに相当する。F0が近いだけで自然とは限らないが、話し方や抑揚の違和感を見るうえで重要である。

フォルマントは、声道の共鳴周波数である。特にF1/F2/F3は母音の響きに関係する。本人らしい母音の響きが出ているかを見る補助指標になる。

DTWはDynamic Time Warpingの略で、少し速く読んだ音声と少し遅く読んだ音声を、時間方向に伸縮させて比較する手法である。同じ文でもモデルごとに長さが違うため、単純に同じ時刻同士を比べるよりDTWのほうが現実に合う。

数値は、次のように読むと分かりやすい。

| 値の種類 | 例 | 読み方 |
|---|---|---|
| 0から1のsimilarity | `Composite 0.6472`、`Prosody 0.7710` | 1に近いほど元音声に近い。ただし0.01未満の差は誤差や条件差の範囲として扱う |
| 比率 | `Duration ratio 0.947` | 1.0に近いほど長さが近い。0.95なら元音声より約5%短く、1.10なら約10%長い |
| 差分 | `Median F0 delta +12Hz` | 0に近いほど近い。プラスなら生成音声のほうが高め、マイナスなら低め |
| 距離 | `Formant relative distance 0.06` | 低いほど近い。similarityとは逆で、値が小さいほうが良い |
| 相関 | `F0 corr 0.80` | 高いほど上下の動きが似ている。ただし声の高さの中心がずれていても相関は高くなり得る |
| `nan` | `F1 delta nan` | 測定不能という意味。無音、短すぎる区間、F0やフォルマントを安定推定できない区間で出る |

重要なのは、1つの数値で声の良し悪しを決めないことだ。例えば、ASR一致率が高い音声は本文を正しく読めている可能性が高いが、声質が本人に似ているとは限らない。F0が近い音声は高さの中心が近いが、声色や発音が近いとは限らない。フォルマントが近い音声は母音の響きが近い可能性があるが、子音の出方や語尾の自然さまでは説明しない。

## 同一文による音響比較

評価対象には、参照音声に使っていないH4n録音4文を選んだ。各モデルには、元録音と同じ文章を読ませた。

| Sample | Category | Text |
|---|---|---|
| `similarity_h4n_005` | short_explanation | はい、では次に、数字と固有名詞を含む文章をもう一度確認しましょう。 |
| `similarity_h4n_026` | technical_terms | RTX 5060 Ti、VRAM 16GB、CUDA、Python、PyTorch を確認します。 |
| `similarity_h4n_051` | narration | これは説明動画の冒頭で使う、落ち着いた読み上げの文章です。 |
| `similarity_h4n_087` | contextual_reading | 生き物の声と、生ビールの注文では、生の読み方が変わります。 |

`similarity_h4n_026` は意図的に難しい文である。英字、製品名、数字、技術語が混ざると、TTSもASRも崩れやすい。

総合結果は以下。

| Model | Composite | Timbre | Frame/DTW | Prosody | Formants | Duration ratio |
|---|---:|---:|---:|---:|---:|---:|
| Irodori-TTS 500M | 0.6305 | 0.9745 | 0.1539 | 0.6824 | 0.8787 | 1.041 |
| Irodori-TTS 600M VoiceDesign | 0.6472 | 0.9643 | 0.1591 | 0.7710 | 0.8273 | 1.041 |
| Qwen3-TTS 1.7B | 0.6382 | 0.9602 | 0.1611 | 0.7328 | 0.8111 | 0.853 |
| Fish Speech S2 Pro | 0.6472 | 0.9684 | 0.1734 | 0.7387 | 0.8452 | 0.862 |
| CosyVoice2 | 0.6365 | 0.9554 | 0.1693 | 0.7028 | 0.7818 | 0.947 |
| VoxCPM2 ultimate | 0.6486 | 0.9638 | 0.1801 | 0.7264 | 0.8239 | 0.918 |

![Overall model similarity](/images/local-japanese-tts-voice-clone/acoustic_similarity_deep/model_overview.png)

Compositeだけを見ると、VoxCPM2 ultimate、Irodori-TTS 600M VoiceDesign、Fish Speech S2 Proがほぼ同点のトップ群である。差は非常に小さいため、数値上の1位だけを強調するより、聴感評価と合わせて読むべきである。

ProsodyはIrodori-TTS 600M VoiceDesignが最も高い。これは、VoiceDesignのcaption制御が抑揚に効いている可能性を示している。

Frame/DTWはVoxCPM2 ultimateが最も高い。これは、同じ文を読んだときの音響特徴の時間的な並びが比較的近いことを示す。VoxCPM2は内容保持が強いが、抑揚の自然さでは聴感確認が必要だった。

Fish SpeechはCompositeでは同点トップ群で、聴感では最も自然だった。数値だけで説明しきれない自然さがあり、最終品質の基準として使いやすい。

## ASR内容保持

ASR一致率は以下。

| Model | Count | Review | Avg Ratio | Avg Duration |
|---|---:|---:|---:|---:|
| Irodori-TTS 500M | 4 | 1 | 0.7807 | 6.43s |
| Irodori-TTS 600M VoiceDesign | 4 | 1 | 0.7790 | 6.27s |
| Qwen3-TTS 1.7B | 4 | 1 | 0.7708 | 5.06s |
| Fish Speech S2 Pro | 4 | 1 | 0.7820 | 5.05s |
| CosyVoice2 | 4 | 1 | 0.7556 | 5.98s |
| VoxCPM2 ultimate | 4 | 1 | 0.8179 | 5.56s |

Avg Ratioは高いほど、期待テキストとASR文字起こしが近い。VoxCPM2 ultimateが最も高い。ただし、これは「声が本人に似ている」という意味ではない。あくまで本文を保てているかの指標である。

全モデルでReviewが1本ある。主な原因は `similarity_h4n_026` の技術語である。例えば、`RTX 5060 Ti`、`VRAM 16GB`、`CUDA`、`PyTorch` は、表記のまま読ませると崩れやすい。

技術系ナレーションでは、次のように読みを開いたほうが安定する。

| 表記 | 読み寄せ例 |
|---|---|
| RTX 5060 Ti | アールティーエックス ごーまるろくまる ティーアイ |
| VRAM 16GB | ブイラム じゅうろくギガバイト |
| CUDA | クーダ |
| Python | パイソン |
| PyTorch | パイトーチ |

## ピッチとフォルマント

F0とフォルマント系の結果は以下。

| Model | Median F0 delta | F0 contour sim | F0 corr | Formant relative distance |
|---|---:|---:|---:|---:|
| Irodori-TTS 500M | +17.44 Hz | 0.6824 | 0.7228 | 0.0461 |
| Irodori-TTS 600M VoiceDesign | +0.20 Hz | 0.7710 | 0.8023 | 0.0673 |
| Qwen3-TTS 1.7B | +12.31 Hz | 0.7328 | 0.8043 | 0.0746 |
| Fish Speech S2 Pro | +6.34 Hz | 0.7387 | 0.7693 | 0.0593 |
| CosyVoice2 | +13.44 Hz | 0.7028 | 0.7061 | 0.0872 |
| VoxCPM2 ultimate | +11.41 Hz | 0.7264 | 0.7626 | 0.0681 |

Median F0 deltaは0Hzに近いほど、声の高さの中心が元音声に近い。Irodori-TTS 600M VoiceDesignは+0.20Hzで最も近い。

F0 contour simは高いほど、ピッチの上下の形が近い。これもIrodori-TTS 600M VoiceDesignが最も高い。抑揚制御の面では強い。

Formant relative distanceは低いほど良い。Irodori-TTS 500MとFish Speechはこの値が比較的低く、母音・声道共鳴の近さでは悪くない。ただし、Irodori-TTS 500MはF0が高めに出るため、総合評価ではVoiceDesign版より下になった。

F0を図にすると、局所的な違和感が見えやすい。

![F0 overlay similarity_h4n_087](/images/local-japanese-tts-voice-clone/acoustic_similarity_deep/f0_overlay_similarity_h4n_087.png)

黒線が元のH4n音声で、色付きの線が各モデルである。同じ文を読んでいても、ピッチの山、谷、語尾の落ち方がかなり違う。VoxCPM2は内容保持とDTWは強いが、局所的なピッチ運びにはズレがある。Qwen3-TTSも一部で上方向に跳ねる。

メルスペクトログラムでは、周波数帯ごとのエネルギー分布を見られる。

![Spectrogram grid similarity_h4n_087](/images/local-japanese-tts-voice-clone/acoustic_similarity_deep/spectrogram_grid_similarity_h4n_087.png)

スペクトログラム上では、母音の低域成分、子音の高域ノイズ、無音の入り方、語尾の伸びが見える。音声を聞かなくても、モデルごとの発話長や帯域の違いが確認できる。

## 音素・モーラ単位の追加分析

ここまでの分析は、文全体または発話セグメント単位の比較だった。しかし、日本語TTSの違和感はもっと細かい単位で出る。母音の響き、子音の立ち上がり、撥音、長音、読点の間、語尾のF0下降などである。

そこで追加で、疑似的なモーラ/音素単位の分析を行った。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\analyze_phonetic_micro_similarity.py
```

この分析では、まず各評価文に対して読みを定義した。例えば `similarity_h4n_087` は次のように扱った。

```text
生き物の声と、生ビールの注文では、生の読み方が変わります。
いきもののこえと、なまびーるのちゅうもんでは、なまのよみかたがかわります。
```

次に、読みをモーラ列に分解し、母音核、子音オンセット、破裂音、摩擦音/破擦音、鼻音、撥音、長音、ポーズに分類した。元H4n音声上に近似モーラ境界を作り、各モデル側へはMFCC-DTWで境界を写像した。つまりPraat/TextGridのような厳密な強制アラインメントではないが、同じテキスト位置で音響特徴を比較するための実用的な近似である。

指標の読み方は次のとおり。

| 指標 | 見ているもの | 良い値の読み方 |
|---|---|---|
| Unit similarity | モーラ全体のMFCC、F0、フォルマント、長さ、オンセット | 高いほど、そのモーラが元音声に近い |
| Vowel nucleus | 母音核のMFCC、F1/F2/F3、F0 | 高いほど母音の響きが近い |
| Consonant onset | 子音立ち上がりのMFCC、高域比、ZCR、スペクトル重心 | 高いほど子音の出だしが近い |
| Pause | 読点・句点付近の長さと静けさ | 高いほど間の取り方が近い |

ここで言う「モーラ」は、日本語のリズム上の単位である。例えば `なまびーる` は、おおまかに `な / ま / び / ー / る` のように数えられる。英語のsyllableとは一致しないが、日本語TTSの間、長音、撥音、促音を見るには扱いやすい。

「母音核」は、各モーラの中で母音が安定して鳴っている中心部分である。`か` なら子音 `k` の立ち上がりの後に来る `a` の部分を見る。「子音オンセット」は、その前の子音の出だしで、破裂音なら閉鎖から開放される瞬間、摩擦音なら高域ノイズが立ち上がる部分を指す。

この分析の数値は、0.80なら「80%似ている」と読むものではない。あくまで今回の特徴量と重みで作った比較スコアである。モデル間の傾向、苦手な音の種類、局所的な破綻箇所を探すために使う。

モデル別・音声カテゴリ別の結果は以下。

| Model | All mora | Vowel nucleus | Consonant onset | Plosive | Fric./Affr. | Nasal | Pause |
|---|---:|---:|---:|---:|---:|---:|---:|
| Irodori-TTS 500M | 0.7925 | 0.7922 | 0.8071 | 0.8323 | 0.7829 | 0.7915 | 0.6283 |
| Irodori-TTS 600M VoiceDesign | 0.7860 | 0.7834 | 0.7982 | 0.8072 | 0.7683 | 0.8069 | 0.4538 |
| Qwen3-TTS 1.7B | 0.8080 | 0.8066 | 0.8176 | 0.8203 | 0.8011 | 0.8235 | 0.6508 |
| Fish Speech S2 Pro | 0.7921 | 0.7922 | 0.8069 | 0.8118 | 0.7928 | 0.8091 | 0.5941 |
| CosyVoice2 | 0.8059 | 0.8045 | 0.8167 | 0.8251 | 0.7943 | 0.8231 | 0.6779 |
| VoxCPM2 ultimate | 0.8187 | 0.8168 | 0.8314 | 0.8308 | 0.8176 | 0.8438 | 0.6993 |

![Phone class similarity heatmap](/images/local-japanese-tts-voice-clone/acoustic_phonetic_micro/phone_class_similarity_heatmap.png)

この表では、VoxCPM2 ultimateが多くのカテゴリで高い。これは、前述のFrame/DTWやASR内容保持が高かった結果と整合する。VoxCPM2は同じテキスト位置での音響パターンをかなり保てている。

ただし、この結果は「聴感でもVoxCPM2が一番自然」という意味ではない。子音や母音の局所的な音響距離は近くても、発話全体の抑揚、息遣い、語尾、自然な間の置き方で違和感が残ることがある。聴感でFish Speechが最も自然に聞こえた理由は、こうした局所スコアだけでは説明しきれない、文全体のなめらかさにある。

`similarity_h4n_087` を、元の日本語テキストとモーララベル付きで見るとこうなる。

![Annotated spectrogram similarity_h4n_087](/images/local-japanese-tts-voice-clone/acoustic_phonetic_micro/annotated_spectrogram_similarity_h4n_087.png)

上段が元H4n音声で、下に各モデルのメルスペクトログラムを並べている。`生き物` の `いき`、`生ビール` の `なまびーる`、`読み方` の `よみかた` など、同じテキスト位置の帯域エネルギーを見比べられる。母音は低域の倍音構造、摩擦音・破擦音は高域ノイズ、読点は黒い縦帯として見える。

モーラ単位の類似度ヒートマップでは、どのモデルがどの音で外れたかが分かる。

![Mora-level heatmap similarity_h4n_087](/images/local-japanese-tts-voice-clone/acoustic_phonetic_micro/mora_similarity_heatmap_similarity_h4n_087.png)

例えば `similarity_h4n_087` では、Qwen3-TTSは多くのモーラで高い一方、特定箇所のF0跳ねが聴感上の違和感として残る。Irodori VoiceDesignはF0中央値の一致では強いが、句点付近や一部のポーズで元音声と違う。VoxCPM2はモーラ単位では高いが、局所的なピッチ運びの急さが残る。Fish Speechはこの表だけでは最上位ではないが、スペクトログラム上の連続性と聴感の自然さが良い。

F0をDTWで元音声の時間軸に写像すると、抑揚の違いがさらに見える。

![DTW-warped F0 similarity_h4n_087](/images/local-japanese-tts-voice-clone/acoustic_phonetic_micro/warped_f0_mora_similarity_h4n_087.png)

黒線が元H4n音声である。Qwen3-TTSは一部で上方向に跳ね、CosyVoice2も後半で大きく高くなる箇所がある。VoxCPM2はF0の山谷のタイミングは近いが、曲線が急に動く箇所がある。Irodori VoiceDesignは声の高さの中心はかなり近いが、山の高さが元音声より控えめな箇所がある。

母音については、F1/F2の平均差も見た。

![Vowel formant delta](/images/local-japanese-tts-voice-clone/acoustic_phonetic_micro/vowel_formant_delta.png)

この図は、母音ごとに元H4n音声からのF1/F2差分を置いたものだ。原点に近いほど、その母音の声道共鳴が近い。短いモーラからLPCで推定しているため、Praatで母音核を手動確認した値ほど安定ではないが、「声質が近いようで少し違う」という感覚を説明する補助材料になる。

この追加分析で分かったことは、次のとおり。

- VoxCPM2は、モーラ単位・子音オンセット・母音核の数値では強い。内容保持が強いという印象と一致する。
- Qwen3-TTSは局所的なモーラ類似度は高いが、F0の局所ジャンプが聴感上の不自然さにつながりやすい。
- Irodori VoiceDesignは全体のF0中央値やF0 contourでは良いが、ポーズや一部の長音/母音で元音声と違う。速度と品質のバランスは良いが、完全な本人らしさにはまだ差がある。
- Fish Speechはモーラ単位スコアでは突出しないが、聴感上の自然さでは強い。細部の数値より、文全体の連続性や破綻の少なさが効いている可能性が高い。
- CosyVoice2はH4n参照後にかなり改善しており、子音オンセットやポーズでは悪くない。ただし声質・抑揚の主観評価ではトップ群に一歩届かない。

結論として、音素・モーラ単位の分析は、採用モデルのランキングを単純に置き換えるものではない。むしろ、「どのモデルがどの種類の音で外れるか」を見るための診断である。最終判断は、聴感評価、文全体のF0、ASR内容保持、生成速度と合わせて行う必要がある。

## サンプル別の結果

| Sample | Irodori 500M | Irodori VoiceDesign | Qwen3 | Fish Speech | CosyVoice2 | VoxCPM2 ultimate |
|---|---:|---:|---:|---:|---:|---:|
| `similarity_h4n_005` | 0.6095 | 0.6537 | 0.6415 | 0.6340 | 0.6337 | 0.6374 |
| `similarity_h4n_026` | 0.6529 | 0.6448 | 0.6344 | 0.6623 | 0.6446 | 0.6461 |
| `similarity_h4n_051` | 0.6192 | 0.6306 | 0.6378 | 0.6378 | 0.6304 | 0.6512 |
| `similarity_h4n_087` | 0.6403 | 0.6597 | 0.6388 | 0.6546 | 0.6374 | 0.6596 |

`similarity_h4n_005` は自然な説明文で、Irodori VoiceDesignが最も高い。短い説明文ではcaption制御が効きやすい。

`similarity_h4n_026` は技術語サンプルで、Fish Speechが最も高い。ただしASRでは全モデルが苦戦しており、実用時は読み仮名化が必要である。

`similarity_h4n_051` は短いナレーション文で、VoxCPM2 ultimateが最も高い。長さと音響系列の近さが効いている。

`similarity_h4n_087` は「生き物」「生ビール」「生の読み方」という文脈依存の読みを含む文で、Irodori VoiceDesignとVoxCPM2 ultimateがほぼ同点、Fish Speechが続いた。

## 生成時間

品質だけでなく、生成時間も重要である。

以前の作業ログには、プレゼン生成、同一文生成、長文チャンク生成など、条件の異なるRunが混在していた。そこで、生成時間だけを比較するために、次の3文を別途用意し、各モデルで同じ文章を生成した。

| ID | 種別 | 評価文 |
|---|---|---|
| `genbench_short_01` | 短文 | 今日は、ローカル環境で音声合成の短いテストを行います。 |
| `genbench_narration_01` | ナレーション | この検証では、録音した参照音声を使い、複数のローカルTTSモデルで同じ文章を読み上げさせ、自然さと処理時間を比較します。 |
| `genbench_technical_01` | 技術語 | RTX 5060 Ti、VRAM 16GB、CUDA 12、Python、PyTorchを使い、生成時間とRTFを同じ条件で測定します。 |

計測値は以下である。ここでの生成時間は、各モデルのmanifestに記録した `seconds_elapsed` を使った。Irodori-TTSはCLIが出力する `total_to_decode`、Qwen3-TTS/VoxCPM2/CosyVoice2はスクリプト内で各サンプル生成区間を計測した値である。Fish Speech S2 ProはCLIをサンプルごとに起動し、semantic生成とcodec decodeを含む1サンプル実行全体を測った。

| Model | 生成数 | 生成音声の合計 | 生成時間 | 平均RTF | コメント |
|---|---:|---:|---:|---:|---|
| Irodori-TTS 500M | 3 | 29.9s | 3.5s | 0.12x | captionなし。推論部は非常に速い |
| Irodori-TTS 600M VoiceDesign | 3 | 29.0s | 3.8s | 0.13x | caption制御ありでも高速 |
| CosyVoice2 | 3 | 28.0s | 24.9s | 0.89x | 実時間よりやや速い |
| VoxCPM2 ultimate | 3 | 23.8s | 33.2s | 1.39x | 速いがIrodori/CosyVoice2より重い |
| Qwen3-TTS 1.7B | 3 | 24.6s | 47.0s | 1.91x | 短文なら実用範囲 |
| Fish Speech S2 Pro | 3 | 24.1s | 2994.0s | 124.46x | 品質は強いが速度は明確な外れ値 |

Fish Speech S2 Proも同じ3文で完走させた。3文合計24.1秒の生成音声に対して、生成時間は2994.0秒、平均RTFは124.46xだった。既存のH4nプレゼン比較でも、約48.8秒の音声生成に5253.3秒かかっており、速度面では外れ値として扱うのが妥当である。

下の横棒グラフは平均RTFで並べた。短いほど速い。Irodori-TTSが0.1x台、CosyVoice2が1.0x未満、VoxCPM2とQwen3-TTSが1.0x超、Fish Speechが100x超という関係になった。

![生成時間の比較: RTF目安で昇順、短いほど速い](/images/local-japanese-tts-voice-clone/generation_time_rtf_bar.png)

RTFはReal-Time Factorで、生成時間を音声長で割った値である。RTF 1.0なら1分の音声を約1分で生成する。RTF 2.0なら1分の音声生成に約2分かかる。

なお、Irodori-TTSは今回CLIを1出力ごとに起動して測ったため、コマンド起動込みのwall timeでは500Mが約39.5秒、VoiceDesignが約44.8秒だった。この場合でも平均RTFはそれぞれ約1.32x、約1.55xで、ローカル検証を回す速度としては扱いやすい。

Fish Speechは自然さでは強いが、RTX 5060 Ti 16GB環境ではかなり遅い。大量に試行するモデルではなく、最終候補の高品質サンプルを少数作るモデルとして使うのが現実的である。

## モデル別の考察

### Fish Speech S2 Pro

聴感では最も自然だった。声のまとまり、語尾、破綻の少なさ、プレゼン音声としての聞きやすさが強い。

音響指標でもComposite 0.6472でトップ群に入った。Frame/DTWは0.1734で、Irodori系やQwen3より高い。技術語サンプル `similarity_h4n_026` ではComposite 0.6623で最高だった。

弱点は生成速度である。既存のH4nプレゼン比較では、約49秒の音声生成に約87.6分かかった。今回の同条件3文ベンチマークでも、24.1秒の生成音声に2994.0秒、平均RTF 124.46xかかった。日常的に多数のパラメータを振る用途には重い。

### Irodori-TTS 600M VoiceDesign

実用バランスが最も良い。Compositeは0.6472でFish Speechと同値、Prosodyは0.7710で最高だった。Median F0 deltaも+0.20Hzで、声の高さの中心が元音声に最も近い。

VoiceDesignのcaptionで、落ち着いた声、技術説明向けの明瞭な声、力強い発表口調などを制御できる。声質の完全一致ではFish Speechに及ばないが、同条件3文ベンチマークでは推論部の平均RTFが0.13xで、生成速度、安定性、日本語向けの扱いやすさが強い。

### VoxCPM2 ultimate

音響Compositeは0.6486で最高、ASR Avg Ratioも0.8179で最高だった。内容保持と時系列特徴の近さでは非常に強い。

一方で、聴感では抑揚が少し不自然に聞こえる。F0 contour simは0.7264で、Irodori VoiceDesignやFish Speechより低い。つまり、文字内容や音響系列は近いが、ピッチの運び方で本人らしさから外れる箇所がある。

短文ナレーションや内容保持重視の用途では候補に戻せる。採用するなら、cfgやtimestepsを振って抑揚のクセを抑えられるかを見る価値がある。

### Qwen3-TTS 1.7B

短文チャンクでは十分良い。Compositeは0.6382で中位、F0 corrは0.8043と高い。

ただし、Median F0 deltaが+12.31Hzで、元音声より少し高めに出る。局所的なピッチ跳ねもあり、イントネーションの違和感につながる。技術語、数字、URL、長文をまとめて渡すと崩れやすい。

使うなら、短文チャンク、読み正規化、token cap管理を前提にしたほうがよい。

### CosyVoice2

H4n Pro参照音声ではかなり改善した。Compositeは0.6365で、Qwen3-TTSやIrodori 500Mに近い。

Duration ratioは0.947で、話速や長さは比較的近い。一方、Prosody 0.7028、Formants 0.7818はトップ群より低い。声質・抑揚の両面で少し外れる。

生成時に、target textがprompt textより短いという警告が出た。参照文と生成文の長さ・内容バランスを調整すると、まだ改善余地がある。

### Irodori-TTS 500M

軽く扱いやすい。同条件3文ベンチマークでは推論部の平均RTFが0.12xで最速だった。Timbre 0.9745とFormants 0.8787は高く、音色系の特徴も悪くない。

一方、Prosody 0.6824、Median F0 delta +17.44Hzが弱い。本人声に寄せる目的では、600M VoiceDesignを優先したい。

### F5-TTS、GPT-SoVITS、Style-Bert-VITS2

F5-TTSは、日本語生成として中国語寄りに聞こえ、本稿の用途では不採用とした。

GPT-SoVITSは、音質にノイズ感があり、聴感評価で厳しかった。

Style-Bert-VITS2は、デフォルト推論は動いた。ユーザー音声100本からデータセット作成、BERT特徴量、style vector生成までは進んだが、学習は安定して完了しなかった。RTX 50系、Torch、CUDAの組み合わせに起因する可能性があり、公式推奨環境に寄せた別環境で再試行するのがよさそうだった。

## 長めのプレゼン音声

短文だけでなく、Irodori-TTS 600M VoiceDesignで架空のEV車両の新車発表会スピーチも生成した。ひとつの長文をそのまま読ませるのではなく、40個の短いチャンクに分け、各チャンクにVoiceDesignのcaptionを付けた。

manifestの例は以下。

```tsv
id	tone	duration_scale	text_path	caption
001	calm_opening	1.05	samples/text/ev_launch_irodori_600m/ev_launch_chunk_001.txt	落ち着いた成人男性の声。大きな会場の冒頭で、近い距離感を残しながら、丁寧で安定したトーンで話す。
004	confident_reveal	1.00	samples/text/ev_launch_irodori_600m/ev_launch_chunk_004.txt	新製品を発表する自信のある声。明瞭で、語尾を力強くまとめる。
013	battery_precise	1.00	samples/text/ev_launch_irodori_600m/ev_launch_chunk_013.txt	数字と性能を正確に伝える声。落ち着いて、少し硬めに、誤解なく読み上げる。
028	heartfelt_closing	1.08	samples/text/ev_launch_irodori_600m/ev_launch_chunk_028.txt	心を込めた締めの声。感謝を含み、穏やかで余韻を残す。
```

生成コマンドは以下。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\irodori_voicedesign_manifest_generate.ps1 `
  -ManifestPath samples\manifests\irodori_ev_launch_manifest.tsv `
  -RefAudio samples\voice_refs\h4n_ref20s_neutral_091_094.wav `
  -RefLabel h4n_ref20s_neutral_091_094 `
  -OutputDir outputs\irodori_ev_launch\chunks `
  -NumSteps 24 `
  -CfgScaleCaption 3.5
```

生成後、-3 dBFSにピーク正規化して結合した。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\normalize_wav_peak.py `
  --input-glob "outputs/irodori_ev_launch/chunks/*.wav" `
  --output-dir outputs\irodori_ev_launch\normalized_minus3db_chunks `
  --peak-dbfs -3

tools\voxcpm\.venv\Scripts\python.exe scripts\concat_wav_chunks.py `
  --output outputs\irodori_ev_launch\irodori_ev_launch_600m_voicedesign_5min_normalized.wav `
  --silence-ms 350 `
  --take-first 30 `
  --glob "outputs/irodori_ev_launch/normalized_minus3db_chunks/*.wav"
```

結果は以下。

| 出力 | 長さ | Peak | Clips |
|---|---:|---:|---:|
| `irodori_ev_launch_600m_voicedesign_5min_normalized.wav` | 299.230s | -3.00 dBFS | 0 |
| `irodori_ev_launch_600m_voicedesign_extended_normalized.wav` | 401.250s | -3.00 dBFS | 0 |

40チャンク全体のASR一致率は0.9590で、Reviewは0だった。長めのプレゼン音声では、Irodori-TTS 600M VoiceDesignがかなり扱いやすい。

## 日本語TTSで効いた工夫

### 長文をそのまま入れない

長い文章を一気にTTSへ渡すと、途中で破綻しやすい。特にQwen3-TTSでは、長文、数字、英語、URL、長い参照音声が重なると崩れた。

実用上は、1チャンクを数秒から十数秒に分け、生成後に300msから400ms程度の無音を挟んで結合するのが安定した。

### 読みを正規化する

日本語では、表記と読みが一致しない。技術語、数字、日付、文脈依存の漢字は、モデル任せにしないほうがよい。

| 表記 | 渡したい読みの例 |
|---|---|
| 7日 | なのか |
| 9日 | ここのか |
| 生 | なま / せい / しょう |
| 404 | よんまるよん |
| 500 | ごまるまる |
| GPU | ジーピーユー |
| URL | ユーアールエル |
| 10% | じゅっパーセント |

読みを正規化すると、ASR一致率だけでなく、聴感上の違和感も減る。

### 参照音声を選ぶ

参照音声は、長さより品質が重要だった。ノイズが少なく、普通のテンションで、言い直しがなく、音量が安定した8秒から20秒程度の音声が使いやすい。

抑揚の強い音声、長すぎる音声、文脈が特殊な音声は、モデルによってはかえって不安定になる。

### 音量は後処理で整える

生成音声はモデルごとに音量がばらつく。聴き比べや公開用サンプルでは、-3 dBFS程度にピーク正規化すると扱いやすい。

```powershell
tools\voxcpm\.venv\Scripts\python.exe scripts\normalize_wav_peak.py `
  --input-glob "outputs/irodori_ev_launch/chunks/*.wav" `
  --output-dir outputs\irodori_ev_launch\normalized_minus3db_chunks `
  --peak-dbfs -3
```

### ASRだけで勝敗を決めない

ASR一致率は便利だが、TTS品質そのものではない。VoxCPM2のように内容保持が強くても抑揚に違和感が残るモデルがある。逆に、Fish Speechのように数値上は僅差でも、聴感では自然さが際立つモデルもある。

最終判断では、最低でも次を分けて見る必要がある。

- 本文を正しく読めているか
- 声質が本人に近いか
- 抑揚が自然か
- 長文で破綻しないか
- 生成速度が運用に耐えるか

## 結論

ローカルPCだけでも、日本語の本人声TTSはかなり試せる。ただし、モデルを入れれば終わりではない。録音品質、読み正規化、チャンク分割、参照音声の選び方、後処理、評価設計が結果を大きく左右する。

本検証の結論は以下。

| 用途 | 選ぶモデル |
|---|---|
| 最高品質のサンプルを少数作る | Fish Speech S2 Pro |
| 速度と品質のバランスを重視する | Irodori-TTS 600M VoiceDesign |
| 内容保持と短文ナレーションを重視する | VoxCPM2 ultimate |
| 条件調整込みで再検討する | CosyVoice2 |
| 短文を細かく制御して試す | Qwen3-TTS 1.7B |
| 本格的な自分専用モデル学習 | Style-Bert-VITS2などを公式推奨環境で再試行 |

実用的には、Irodori-TTS 600M VoiceDesignで高速に原稿、読み、参照音声を詰め、最終品質のサンプルをFish Speechで少数作るのが扱いやすい。内容保持を重視する短文ではVoxCPM2も候補に入る。

技術語や数字を含む日本語TTSでは、モデル選びだけでなく、モデルが壊れにくい入力を作ることが重要だった。

## 補足資料

公開用の補足資料、分析TSV、試聴ページ、公開可能なスクリプトはGitHubリポジトリにまとめた。プライバシー保護のため、本人の元録音、フル生成音声、モデル重み、ローカル作業ログは含めていない。

- GitHub Pages 試聴ページ: [https://jacky-fuji.github.io/mytts-public/](https://jacky-fuji.github.io/mytts-public/)
- Zenn公開記事: [https://zenn.dev/fujinumagic/articles/local-japanese-tts-voice-clone](https://zenn.dev/fujinumagic/articles/local-japanese-tts-voice-clone)
- 公開リポジトリ: [jacky-fuji/mytts-public](https://github.com/jacky-fuji/mytts-public)
- Zenn記事Markdown: [`articles/local-japanese-tts-voice-clone.md`](https://github.com/jacky-fuji/mytts-public/blob/main/articles/local-japanese-tts-voice-clone.md)
- 録音スクリプト100文: [`data/recording_script_ja_100.txt`](https://github.com/jacky-fuji/mytts-public/blob/main/data/recording_script_ja_100.txt)
- 同一文評価ターゲット: [`data/similarity_eval_targets.tsv`](https://github.com/jacky-fuji/mytts-public/blob/main/data/similarity_eval_targets.tsv)
- 生成時間ベンチマーク文: [`data/generation_time_benchmark_targets.tsv`](https://github.com/jacky-fuji/mytts-public/blob/main/data/generation_time_benchmark_targets.tsv)
- 生成時間ベンチマーク結果: [`data/metrics/generation_time_benchmark_summary.tsv`](https://github.com/jacky-fuji/mytts-public/blob/main/data/metrics/generation_time_benchmark_summary.tsv)
- 評価メトリクス: [`data/metrics/`](https://github.com/jacky-fuji/mytts-public/tree/main/data/metrics)
- 分析・図表生成スクリプト: [`scripts/`](https://github.com/jacky-fuji/mytts-public/tree/main/scripts)
- GitHub Pages用HTMLと公開MP3: [`docs/`](https://github.com/jacky-fuji/mytts-public/tree/main/docs)
