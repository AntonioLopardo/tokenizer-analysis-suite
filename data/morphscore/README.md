---
license: cc-by-sa-4.0
language:
- ar
- en
- de
- ru
- tr
- ab
- af
- sq
- am
- az
- bm
- eu
- be
- bn
- br
- bg
- ca
- zh
- hr
- cs
- da
- nl
- et
- fi
- fr
- ceb
- bur
- myv
- gl
- ka
- el
- gu
- ht
- he
- hi
- hu
- is
- id
- ga
- it
- ja
- kk
- kpv
- ko
- ky
- lv
- lt
- mk
- ml
- gv
- mr
- mdf
- 'no'
- oc
- ps
- fa
- pl
- pt
- ro
- sa
- gd
- se
- sd
- si
- sk
- sl
- es
- sr
- tl
- ta
- tt
- uk
- ur
- ug
- uz
- vep
- vi
- cy
- wo
- sah
- ltg
- lij
- hsb
size_categories:
- 1M<n<10M
---

# MorphScore

MorphScore is a tokenizer evaluation framework, which evaluates the extent to which a tokenizer segments words along morpheme boundaries. This repository contains the datasets used to calculate MorphScore.
In total, we have datasetes for 86 languages, but only 70 languages have at least 100 items after filtering. 

All datasets are derived from existing [Universal Dependencies treebanks](https://universaldependencies.org/). In the table below, we link the source dataset for each language. 

See the [new preprint](https://arxiv.org/abs/2507.06378) for details and see the [GitHub repository](https://github.com/catherinearnett/morphscore) for details on using MorphScore.


## Languages

| **Language** | **ISO 639-3** | **ISO 15924** | **Num. Words** | **Num. Unique Words** | **Data Source(s)** | 
|--------------|---------------|---------------|-----------------|-----------------------|---------------------|
| Abkhaz       | abk           | cyrl          | 4518            | 3222                  | [UD_Abkhaz-AbNC](https://github.com/UniversalDependencies/UD_Abkhaz-AbNC/tree/master) v2.16 |
| Afrikaans    | afr           | latn          | 7011            | 2093                  | [UD_Afrikaans-AfriBooms](https://github.com/UniversalDependencies/UD_Afrikaans-AfriBooms/tree/master) v2.16 |
| Albanian     | sqi (tos)     | latn          | 897             | 702                   | [UD_Albanian-STAF](https://github.com/UniversalDependencies/UD_Albanian-STAF) v2.15 |
| Amharic      | amh           | ethi          | 39              | 9                     | [UD_Amharic-ATT](https://github.com/UniversalDependencies/UD_Amharic-ATT) v2.11 |
| Armenian     | hye           | armn          | 18906           | 8994                  | [UD_Armenian-ArmTDP](https://github.com/UniversalDependencies/UD_Armenian-ArmTDP/tree/master) v2.13 (except test = v2.10) |
| Azerbaijani  | aze           | latn          | 368             | 251                   | [UD_Azerbaijani-TueCL](https://github.com/UniversalDependencies/UD_Azerbaijani-TueCL) v2.16 |
| Bambara      | bam           | latn          | 6849            | 665                   | [UD_Bambara-CRB](https://github.com/UniversalDependencies/UD_Bambara-CRB) v2.13 |
| Basque       | eus           | latn          | 50901           | 15932                 | [UD_Basque-BDT](https://github.com/UniversalDependencies/UD_Basque-BDT) v2.14 |
| Belarusian   | bel           | cyrl          | 104096          | 31129                 | [UD_Belarusian-HSE](https://github.com/UniversalDependencies/UD_Belarusian-HSE/tree/master) v2.16 |
| Bengali      | ben           | beng          | 102             | 67                    | [UD_Bengali-BRU](https://github.com/UniversalDependencies/UD_Bengali-BRU) v2.11 |
| Bhojpuri     | bho           | deva          | 1371            | 360                   | [UD_Bhojpuri-BHTB](https://github.com/UniversalDependencies/UD_Bhojpuri-BHTB) v2.16 |
| Breton       | bre           | latn          | 3054            | 1021                  | [UD_Breton-KEB](https://github.com/UniversalDependencies/UD_Breton-KEB) v2.12 |
| Bulgarian    | bul           | cyrl          | 44412           | 15468                 | [UD_Bulgarian-BTB](https://github.com/UniversalDependencies/UD_Bulgarian-BTB) 2.16 |
| Buriat/Buryat| bur           | cyrl          | 3646            | 2322                  | [UD_Buryat-BDT](https://github.com/UniversalDependencies/UD_Buryat-BDT) v2.11 |
| Cantonese/Yue Chinese| yue   | hant          | 42              | 13                    | [UD_Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK) v2.12 |
| Catalan      | cat           | latn          | 15495           | 3385                  | [UD_Catalan-AnCora](https://github.com/UniversalDependencies/UD_Catalan-AnCora) v2.15 |
| Cebuano      | ceb           | latn          | 264             | 136                   | [UD_Cebuano-GJA](https://github.com/UniversalDependencies/UD_Cebuano-GJA) v2.15 |
| Mandarin Chinese  | zho/cmn  | hans          | 915             | 140                   | [UD_Chinese-GSDSimp](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp) v2.13 |
| Croatian     | hrv           | latn          | 75681           | 23363                 | [UD_Croatian-SET](https://github.com/UniversalDependencies/UD_Croatian-SET) v2.11 (except train = v2.13) |
| Czech        | ces           | latn          | 200387          | 44025                 | [UD_Czech-CAC](https://github.com/UniversalDependencies/UD_Czech-CAC/tree/master) v2.16 |
| Danish       | dan           | latn          | 25662           | 8256                  | [UD_Danish-DDT](https://github.com/UniversalDependencies/UD_Danish-DDT) v2.14 |
| Dutch        | nld           | latn          | 42900           | 12512                 | [UD_Dutch-Alpino](https://github.com/UniversalDependencies/UD_Dutch-Alpino/tree/master) v2.14 |
| English      | eng           | latn          | 31039           | 5844                  | [UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT/tree/master) v2.16 |
| Erzya        | myv           | cyrl          | 8147            | 4826                  | [UD_Erzya-JR](https://github.com/UniversalDependencies/UD_Erzya-JR) v2.16|
| Estonian     | est           | latn          | 199597          | 62170                 | [UD_Estonian-EDT](https://github.com/UniversalDependencies/UD_Estonian-EDT) v2.16 |
| Finnish      | fin           | latn          | 99308           | 41095                 | [UD_Finnish-TDT](https://github.com/UniversalDependencies/UD_Finnish-TDT/tree/master) v2.16|
| French       | fra           | latn          | 103887          | 13508                 | [UD_French-GSD](https://github.com/UniversalDependencies/UD_French-GSD) v2.16 |
| Galician     | glg           | latn          | 29484           | 7113                  | [UD_Galician-CTG](https://github.com/UniversalDependencies/UD_Galician-CTG) v2.12 |
| Georgian     | kat           | geor          | 20148           | 8622                  | [UD_Georgian-GLC](https://github.com/UniversalDependencies/UD_Georgian-GLC) v2.16 |
| German       | deu           | latn          | 949279          | 49239                 | [UD_German-HDT](https://github.com/UniversalDependencies/UD_German-HDT/tree/master) v2.16 |
| Greek        | ell           | grek          | 21428           | 6986                  | [UD_Greek-GDT](https://github.com/UniversalDependencies/UD_Greek-GDT/tree/master) v2.16 |
| Gujarati     | guj           | gujr          | 120             | 81                    | [UD_Gujarati-GujTB](https://github.com/UniversalDependencies/UD_Gujarati-GujTB/tree/master) v2.14 |
| Haitian Creole| hat          | latn          | 950             | 37                    | [UD_Haitian_Creole-Adolphe](https://github.com/UniversalDependencies/UD_Haitian_Creole-Adolphe) v2.16 |
| Hebrew       | heb           | hebr          | 43102           | 9768                  | [UD_Hebrew-HTB](https://github.com/UniversalDependencies/UD_Hebrew-HTB) v2.15 |
| Hindi        | hin           | deva          | 83865           | 4749                  | [UD_Hindi-HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB) v2.15 |
| Hungarian    | hun           | latn          | 12349           | 8103                  | [UD_Hungarian-Szeged](https://github.com/UniversalDependencies/UD_Hungarian-Szeged) v2.11 |
| Icelandic    | isl           | latn          | 329620          | 40686                 | [UD_Icelandic-IcePaHC](https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC) v2.16|
| Indonesian   | ind           | latn          | 16844           | 3400                  | [UD_Indonesian-GSD](https://github.com/UniversalDependencies/UD_Indonesian-GSD) v2.16 |
| Irish        | gle           | latn          | 31911           | 7474                  | [UD_Irish-IDT](https://github.com/UniversalDependencies/UD_Irish-IDT/tree/master) v2.15 |
| Italian      | ita           | latn          | 73790           | 10981                 | [UD_Italian-ISDT](https://github.com/UniversalDependencies/UD_Italian-ISDT/tree/master) v2.16 |
| Japanese     | jpn           | jpan          | 16898           | 5427                  | [UD_Japanese-GSDLUW](https://github.com/UniversalDependencies/UD_Japanese-GSDLUW) v2.15|
| Kazakh       | kaz           | cyrl          | 4200            | 2738                  | [UD_Kazakh-KTB](https://github.com/UniversalDependencies/UD_Kazakh-KTB) v2.11 |
| Komi-Zyrian  | kpv           | cyrl          | 2670            | 2017                  | [UD_Komi_Zyrian-Lattice](https://github.com/UniversalDependencies/UD_Komi_Zyrian-Lattice) v2.16 |
| Korean       | kor           | hang          | 239862          | 86349                 | [UD_Korean-Kaist](https://github.com/UniversalDependencies/UD_Korean-Kaist) v2.11 |
| Kirghiz/Kyrgyz| kir          | cyrl          | 11752           | 4659                  | [UD_Kyrgyz-KTMU](https://github.com/UniversalDependencies/UD_Kyrgyz-KTMU) v2.15 (train), v2.16 (test) |
| Latvian      | lav           | latn          | 141550          | 40081                 | [UD_Latvian-LVTB](https://github.com/UniversalDependencies/UD_Latvian-LVTB/tree/master) v2.16 |
| Lithuanian   | lit           | latn          | 32378           | 13140                 | [UD_Lithuanian-ALKSNIS](https://github.com/UniversalDependencies/UD_Lithuanian-ALKSNIS) v2.16, except test (v2.15) |
| Macedonian   | mkd           | cyrl          | 285             | 241                   | [UD_Macedonian-MTB](https://github.com/UniversalDependencies/UD_Macedonian-MTB) v2.13 |
| Malayalam    | mal           | mlym          | 867             | 710                   | [UD_Malayalam-UFAL](https://github.com/UniversalDependencies/UD_Malayalam-UFAL) v2.12 |
| Manx         | glv           | latn          | 2299            | 873                   | [UD_Manx-Cadhan](https://github.com/UniversalDependencies/UD_Manx-Cadhan) v2.15 |
| Marathi      | mar           | deva          | 1610            | 663                   | [UD_Marathi-UFAL](https://github.com/UniversalDependencies/UD_Marathi-UFAL) v2.11 |
| Moksha       | mdf           | cyrl          | 1733            | 1401                  | [UD_Moksha-JR](https://github.com/UniversalDependencies/UD_Moksha-JR) v2.16 |
| Northern Sami| sme           | latn          | 10360           | 5024                  | [UD_North_Sami-Giella](https://github.com/UniversalDependencies/UD_North_Sami-Giella) v2.15 (train), v2.12 (test)|
| Norwegian    | nob           | latn          | 87394           | 15716                 | [UD_Norwegian-Bokmaal](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) v2.16|
| Occitan      | oci           | latn          | 6751            | 2561                  | [UD_Occitan-TTB](https://github.com/UniversalDependencies/UD_Occitan-TTB) v2.16 |
| Pashto       | pus           | arab          | 779             | 351                   | [UD_Pashto-Sikaram](https://github.com/UniversalDependencies/UD_Pashto-Sikaram) v2.16|
| Persian      | fas           | arab          | 102374          | 14847                 | [UD_Persian-PerDT](https://github.com/UniversalDependencies/UD_Persian-PerDT) v2.16, except dev (v2.12) |
| Polish       | pol           | latn          | 131836          | 42545                 | [UD_Polish-PDB](https://github.com/UniversalDependencies/UD_Polish-PDB/tree/master) v2.13 |
| Portuguese   | por           | latn          | 69430           | 13074                 | [UD_Portuguese-CINTIL](https://github.com/UniversalDependencies/UD_Portuguese-CINTIL) v2.13 |
| Romanian     | ron           | latn          | 72968           | 19108                 | [UD_Romanian-RRT](https://github.com/UniversalDependencies/UD_Romanian-RRT) v2.13|
| Russian      | rus           | cyrl          | 590060          | 105749                | [UD_Russian-SynTagRus](https://github.com/UniversalDependencies/UD_Russian-SynTagRus/tree/master) v2.16|
| Sanskrit     | san           | deva          | 144583          | 32671                 | [UD_Sanskrit-Vedic](https://github.com/UniversalDependencies/UD_Sanskrit-Vedic) v2.14 |
| Scottish Gaelic | gla        | latn          | 27181           | 3495                  | [UD_Scottish_Gaelic-ARCOSG](https://github.com/UniversalDependencies/UD_Scottish_Gaelic-ARCOSG) v2.16 |
| Serbian      | srp           | latn          | 37324           | 12278                 | [UD_Serbian-SET](https://github.com/UniversalDependencies/UD_Serbian-SET/tree/master) v2.11, except train (v2.13)|
| Sindhi       | snd           | arab          | 3584            | 1132                  | [UD_Sindhi-Isra](https://github.com/UniversalDependencies/UD_Sindhi-Isra/tree/master) v2.16 |
| Sinhala      | sin           | sinh          | 354             | 284                   | [UD_Sinhala-STB](https://github.com/UniversalDependencies/UD_Sinhala-STB) v2.12|
| Slovak       | slk           | latn          | 35545           | 16169                 | [UD_Slovak-SNK](https://github.com/UniversalDependencies/UD_Slovak-SNK) v2.16 |
| Slovenian    | slv           | latn          | 93048           | 32134                 | [UD_Slovenian-SSJ](https://github.com/UniversalDependencies/UD_Slovenian-SSJ) v2.16|
| Spanish      | spa           | latn          | 134449          | 17341                 | [UD_Spanish-AnCora](https://github.com/UniversalDependencies/UD_Spanish-AnCora) v2.15|
| Swedish      | swe           | latn          | 29514           | 8135                  | [UD_Swedish-LinES](https://github.com/UniversalDependencies/UD_Swedish-LinES) v2.16|
| Tagalog      | tgl           | latn          | 212             | 78                    | [UD_Tagalog-Ugnayan](https://github.com/UniversalDependencies/UD_Tagalog-Ugnayan) v2.6|
| Tamil        | tam           | taml          | 4499            | 2178                  | [UD_Tamil-TTB](https://github.com/UniversalDependencies/UD_Tamil-TTB) v2.16 (except dev = v2.13) |
| Tatar        | tat           | cyrl          | 899             | 708                   | [UD_Tatar-NMCTT](https://github.com/UniversalDependencies/UD_Tatar-NMCTT/tree/master) v2.11|
| Turkish      | tur           | latn          | 71984           | 32587                 | [UD_Turkish-Kenet](https://github.com/UniversalDependencies/UD_Turkish-Kenet/tree/master) train (v2.13), dev (v2.10), test (v2.9)|
| Ukrainian    | ukr           | cyrl          | 44288           | 21584                 | [UD_Ukrainian-IU](https://github.com/UniversalDependencies/UD_Ukrainian-IU) v2.11|
| Urdu         | urd           | arab          | 30825           | 2450                  | [UD_Urdu-UDTB](https://github.com/UniversalDependencies/UD_Urdu-UDTB/tree/master) v2.14|
| Uighur/Uyghur| uig           | arab          | 10427           | 3969                  | [UD_Uyghur-UDT](https://github.com/UniversalDependencies/UD_Uyghur-UDT) v2.13|
| Uzbek        | uzb           | latn          | 2466            | 1934                  | [UD_Uzbek-UT](https://github.com/UniversalDependencies/UD_Uzbek-UT) v2.16 |
| Veps         | vep           | latn          | 598             | 399                   | [UD_Veps-VWT](https://github.com/UniversalDependencies/UD_Veps-VWT) v2.16 |
| Vietnamese   | vie           | latn          | 32              | 29                    | [UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB/tree/master) v2.11 (except dev = v2.12)|
| Welsh        | cym           | latn          | 11781           | 2860                  | [UD_Welsh-CCG](https://github.com/UniversalDependencies/UD_Welsh-CCG) v2.16 |
| Wolof        | wol           | latn          | 9211            | 1734                  | [UD_Wolof-WTB](https://github.com/UniversalDependencies/UD_Wolof-WTB/tree/master) v2.13|
| Yakut        | sah           | cyrl          | 479             | 330                   | [UD_Yakut-YKTDT](https://github.com/UniversalDependencies/UD_Yakut-YKTDT) v2.12|
| Latgalian    | ltg           | latn          | 69              | 63                    | [UD_Latgalian-Cairo](https://github.com/UniversalDependencies/UD_Latgalian-Cairo) v2.16|
| Ligurian     | lij           | latn          | 1318            | 637                   | [UD_Ligurian-GLT](https://github.com/UniversalDependencies/UD_Ligurian-GLT/tree/master) train (v2.9), test (v2.16) |
| Upper Sorbian| hsb           | latn          | 3797            | 2498                  | [UD_Upper_Sorbian-UFAL](https://github.com/UniversalDependencies/UD_Upper_Sorbian-UFAL) train (v2.3), test (v2.8)|


## How to Cite

```
@inproceedings{arnett2025alignment,
  author = {Arnett, Catherine and Hudspeth, Marisa and O'Connor, Brendan},
  title = {{Evaluating Morphological Alignment of Tokenizers in 70 Languages}},
  year = {2025},
  booktitle={Proceedings of the ICML 2025 Tokenization Workshop (TokShop)},
  url = {https://arxiv.org/abs/2507.06378}
}


@inproceedings{arnett-bergen-2025-language,
    title = "Why do language models perform worse for morphologically complex languages?",
    author = "Arnett, Catherine  and
      Bergen, Benjamin",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.441/",
    pages = "6607--6623"
}
```