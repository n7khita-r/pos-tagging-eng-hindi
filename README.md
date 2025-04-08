================================== ABOUT ===========================================================================

This is an implementation of HMMs, CRFs, LSTMs and BiLSTMs for POS-tagging done as part of a course assignment. It is a study of
how effective the above models are in POS-tagging for Hindi and English.
It consists of all 4 models, training data (from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5787), manually annotated testing data, and results along with confusion matrices.

================================== IMPLEMENTATION ==============================================

1. Hindi and English datasets have been tagged with a comprehensive list of BIS-POS tags as available in the documentation. These are in the files "eng.txt" and "hin.txt"
2. Hindi and English datasets have been tagged again, but this time a few BIS-POS tags that come under the same category (eg: personal, reflexive, etc. pronouns all considered pronouns, finite, infinite, cardinal all quantities tagged as QT). This has been done for ease of mapping to UD training datasets, which have much less variety compared to
BIS-POS.
3. UD training datasets have been mapped to this set of reduced BIS-POS tags.
4. HMM and CRF models have been trained on this data and used, results are saved.
5. LSTM and BiLSTM models directly output results onto terminal when run 

===================================== DIRECTORY STRUCTURE =====================================================

1. 3 DATASETS FOR ENGLISH AND 3 DATASETS FOR ENGLISH

hin.txt: contains manually annotated data in BIS-POS format
hindi_test.txt: contains manually annotated testing data in **reduced** BIS-POS tags. Some tags have been generalised
to better map with UD tags
hindi_train.txt: contains testing data

eng.txt: contains manually annotated data in BIS-POS format
english_test.txt: contains manually annotated testing data in **reduced** BIS-POS tags. Some tags have been generalised
to better map with UD tags
english_train.txt: contains testing data

2. 3 IMAGES FOR ENGLISH AND 3 IMAGES FOR HINDI SHOWING TAG DISTRIBUTION IN TEST AND TRAINING DATA

hin.png: contains manually annotated data in BIS-POS format (hin.txt)
hindi_test.png: contains manually annotated testing data in **reduced** BIS-POS tags. Some tags have been generalised
to better map with UD tags (hindi_test.txt)
hindi_training.png: contains testing data (hindi_train.txt)

eng.png: contains manually annotated data in BIS-POS format (eng.txt)
english_test.png: contains manually annotated testing data in **reduced** BIS-POS tags. Some tags have been generalised
to better map with UD tags (english_test.txt)
english_training.png: contains testing data (english_train.txt)

3. english_models and hindi_models, english_results and hindi_results

4. hmm_crf.py: the model, count_frequencies.py: for plotting bar graphs against zipf's law

5. training_datasets: contains raw datasets for english, hindi training and testing

6. Confusion matrices for all 4 implementations

======================================== DATASET CITATION =========================================

Zeman, Daniel; et al., 2024, 
  Universal Dependencies 2.15, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, 
  http://hdl.handle.net/11234/1-5787.

