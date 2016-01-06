# Akustinio modelio mokymas

Pirminis šaltinis: http://cmusphinx.sourceforge.net/wiki/tutorialam

Aprašas parengtas naudojant: https://github.com/mondhs/lt-pocketsphinx-tutorial/tree/master/training/liepa


# Reikalavimai


* Linux arba Windows
* Turėti kompiuteryje:
  *  mercurial SCM(http://mercurial.selenic.com/) ir turėti bazines naudojimo žinias.
  * perl .pvz ActivePerl(http://www.activestate.com/activeperl) 
  * python .pvz ActivePython(http://www.activestate.com/activepython)
* Turėti Sphinx programinę įranga:
  * sphinxbase - bilbioteka bendrom atpažinimo funkcijom 
  * pocketsphinx - atpažintuvas
  * SphinxTrain - mokymo biblioteka

# Pasiruošimas apmokymui

* Nusiklonuokite mokyno repositoriją [[http://<Nurodyti Serverį>]]

Pagrindinės apmokymui direktorijos:

* training/liepa/etc - konfigūraciniai failai
* training/liepa/tool - apmokymo paruošimo įrankiai
* training/liepa/wav - garsynai(dėl dydžio ištrinti) ir sakinių transkripcija txt formatu.
  * training/liepa/wav/S003Aa - S003Aa diktoriaus direktorija. Rankomis kurti nereikia.
  * training/liepa/wav/S003Aa/001_01-S003Aa.wav - Sakinio garso failas. S003Aa diktoriaus kodas, 001_01 sakinio unikalus kodas.Rankomis kurti nereikia.
* training/liepa/wav22 - turi būti saugomi mswav 16bit 22kHz formatu. skriptai iš šio katalogo transformuos į 16kHz 16bit formatą, kuris tinka Sphinx.Rankomis kurti nereikia.
  * training/liepa/wav22/D00 - Diktoriaus direktorija. Rankomis kurti nereikia.
  * training/liepa/wav22/D00/S00 - Sakinių direktorija. Rankomis kurti nereikia.
  * training/liepa/wav22/D00/S00/S003Aa_001_01.wav - Sakinio garso failas. S003Aa diktoriaus kodas, 001_01 sakinio unikalus kodas. Rankomis kurti nereikia.
* training/liepa/target - Laikini skaičiavimo failai scriptų. Reikia sukurti rankomis

## Garsyno Sphinx paruošimo procedūra

* Pakeiskite absoliutų kelią iki apmokinimo direktorijos konfigūraciniame faile training/liepa/etc/sphinx_train.cfg. $CFG_BASE_DIR=$SOURCE_DIR/lt-pocketsphinx-tutorial/training/lt vietoj $SOURCE_DIR turėtumete įrašyti kur nesiklonavote repozitorija.
* Pakeiskite absoliutų kelią iki wav22 direktorijos skript faile training/liepa/tool/01_transform_files.py. src_dir = "<CORPUS_DIR>" vietoj turėtumete įrašyti kur yra garsynas diske.
* Sukurkite klaidų loginimo direktoriją /tmp/liepa/transform_files.log. jei norite logus rašyti kitur skript faile training/liepa/tool/01_transform_files.py pakeiskite logging.basicConfig(filename='/tmp/liepa/transform_files.log',level=logging.DEBUG)
* Scriptas pakeis patikrins ar nėra klaidų struktūroje, katalogų struktūra, failų vardūs ir kvandavimo dažnį iš 22kHz į 16kHz. Naudojama sox bilioteka
* Paleiksite training/liepa/tool/01_transform_files.py. jei viskas gerai užsipildys wav direktorija
* Paleiskite training/liepa/tool/02_extract_dict.py - skriptas iš failų esančių wav direktorijoje sukonstruos target direktorijoje *.transcription ir *.fileids failus.
  * Skriptas unifikuos kodavimą į utf-8. Bus vykdoma patikra nekorektiškų simbolių, gramatikos klaidos su hunspell biblioteka.
* Paleiskite training/liepa/tool/03_combine_sentences.py - skriptas iš failų esnačių target direkotorijoje sujungs transkripsijos failus ir sukurs apmokymo ir testavimo duomenų aibes _test.transcription, _train.transcription, _test.fileids, _train.fileids
* Paleiskite training/liepa/tool/04_generate_phonemes.py - naudojant transcribe.exe(nėra sukomitinta, reik prašyti atskirai), bus transformuoti visų sakinių žodžiai iš grafemų į fonemas. taip sukuriant žodyną *.dic ir visų fonemų sąrašą: *.phone
* iš target į etc nukopijuokite rankomis failus: liepa_all.transcription, liepa.dic, liepa.phone, liepa_test.fileids, liepa_test.transcription, liepa_train.fileids, liepa_train.transcription.
* Paleiskite scipta kalbos modelio eksperimentui sukurti: training/liepa/etc/languageModel/make_lang_model.sh. jis paims etc/liepa_all.transcription ir sugeneruos etc/liepa.lm.DMP
* Paleiskite skripta start_training.sh. Mokymas prasidėjo

## Parametrai

* Garsynas
```
# Audio waveform and feature file information
$CFG_WAVFILES_DIR = "$CFG_BASE_DIR/wav";
$CFG_WAVFILE_EXTENSION = 'wav';
$CFG_WAVFILE_TYPE = 'mswav'; # one of nist, mswav, raw
```

* Požymių parametrai
```
$CFG_FEATFILES_DIR = "$CFG_BASE_DIR/feat";
$CFG_FEATFILE_EXTENSION = 'mfc';
$CFG_VECTOR_LENGTH = 13;
```

* Tarpinės direktorijos
```
# Variables used in main training of models
$CFG_DICTIONARY     = "$CFG_LIST_DIR/$CFG_DB_NAME.dic";
$CFG_RAWPHONEFILE   = "$CFG_LIST_DIR/$CFG_DB_NAME.phone";
$CFG_FILLERDICT     = "$CFG_LIST_DIR/.${CFG_DB_NAME}.filler";
$CFG_LISTOFFILES    = "$CFG_LIST_DIR/${CFG_DB_NAME}_train.fileids";
$CFG_TRANSCRIPTFILE = "$CFG_LIST_DIR/${CFG_DB_NAME}_train.transcription";
$CFG_FEATPARAMS     = "$CFG_LIST_DIR/.${CFG_DB_NAME}_feat.params";
```

* Modelio parametrai
```
$CFG_HMM_TYPE  = '.semi.'; # PocketSphinx pusiau testiniai
```

```
  $CFG_DIRLABEL = 'semi';
# 4 srautai  PocketSphinx požymių 
  $CFG_FEATURE = "s2_4x";
  $CFG_NUM_STREAMS = 4;
  $CFG_INITIAL_NUM_DENSITIES = 256; # Gali būti  20h audio = 8, 30h audio = 16, 80h audio = 32
  $CFG_FINAL_NUM_DENSITIES = 256;
```


* Susietų būsenų skaičiius (senones)
```
$CFG_N_TIED_STATES = 200;# Gali būti  20h audio = 2000, 30h audio = 4000, 80h audio = 4000(taip pat)
```

* Garso ir Požymių parametrai
```
# Feature extraction parameters
$CFG_WAVFILE_SRATE = 16000.0;
$CFG_NUM_FILT = 25; # For wideband speech it's 25, for telephone 8khz reasonable value is 15
$CFG_LO_FILT = 130; # For telephone 8kHz speech value is 200
$CFG_HI_FILT = 6800; # For telephone 8kHz speech value is 3500
$CFG_TRANSFORM = "dct";
$CFG_LIFTER = "22";
```

* Dekodavimo parametrai
```
$DEC_CFG_DICTIONARY     = "$CFG_BASE_DIR/etc/$CFG_DB_NAME.dic";
$DEC_CFG_FILLERDICT     = "$CFG_BASE_DIR/etc/.${CFG_DB_NAME}.filler";
$DEC_CFG_LISTOFFILES    = "$CFG_BASE_DIR/etc/${CFG_DB_NAME}_test.fileids";
$DEC_CFG_TRANSCRIPTFILE = "$CFG_BASE_DIR/etc/${CFG_DB_NAME}_test.transcription";
$DEC_CFG_RESULT_DIR     = "$CFG_BASE_DIR/result";
$DEC_CFG_PRESULT_DIR     = "$CFG_BASE_DIR/presult";
# This variables, used by the decoder, have to be user defined, and
# may affect the decoder output
$DEC_CFG_LANGUAGEMODEL  = "$CFG_BASE_DIR/etc/${CFG_DB_NAME}.lm.DMP";
$DEC_CFG_LANGUAGEWEIGHT = "10";
$DEC_CFG_BEAMWIDTH = "1e-80";
$DEC_CFG_WORDBEAM = "1e-40";
```
 

# Apmokymas

Detaliau galite rasti informacijos: Training Acoustic Model For CMUSphinx
Paleiskite komandą: 
* Linux: `sphinxtrain run`
* Windows `python ../sphinxtrain/scripts/sphinxtrain run`

# Pastabos
Pasibaigus apmokymams rezultatus galite surasti liepa.html faile. Aš gavau tokius rezultatus:

    SENTENCE ERROR: 41.6% (247/594) WORD ERROR RATE: 10.4% (572/5481)

