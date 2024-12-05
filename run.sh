********************matlab
running command
====================1.fb15k-237

matlab
write_tree_csv(Tree, 'output/tree/freebase/20231207-075638_nhdp_tree_freebase_po_t25_b01_1000.csv')
write_tree_csv(Tree, 'output/tree/freebase/20240426_all_nhdp_tree_freebase_po_t23_b01_1000.csv')

generate the subset with reference labels by reference_data
# command to use under git bash
! python print-tree.py '../nHDP_matlab/output/tree/freebase/20240102_nhdp_tree_freebase_po_t22_b1_1000_172952.csv' 'data/fb15k-237/vocab_po_20240102172952.txt'
python print-tree.py '../nHDP_matlab/output/tree/freebase/20231108_nhdp_tree_freebase_po_25_1000.csv' 'data/fb15k-237/vocab_po_20231106161156.txt'
python print-tree.py '../nHDP_matlab/output/tree/freebase/20231207-075638_nhdp_tree_freebase_po_t25_b01_1000.csv' 'data/fb15k-237/vocab_po_20231207-075638.txt'
python print-tree.py '../nHDP_matlab/output/tree/freebase/20240426_all_nhdp_tree_freebase_po_t23_b01_1000.csv' 'data/fb15k-237/vocab_po_20240426_all.txt'

# command to use under git bash
! python save_tree.py '../nHDP_matlab/output/tree/freebase/20240102_nhdp_tree_freebase_po_t22_b1_1000_172952.csv' 'data/fb15k-237/vocab_po_20240102172952.txt' 'output/test/20240102_nhdp_tree_freebase_po_t22_b1_1000_172952.csv'
python save_tree.py '../nHDP_matlab/output/tree/freebase/20240102_nhdp_tree_freebase_po_t22_b01_1000_172952.csv' 'data/fb15k-237/vocab_po_20240102172952.txt' 'output/tree/freebase/20240102_nhdp_tree_freebase_po_t22_b01_1000_172952.csv'
python save_tree.py '../nHDP_matlab/output/tree/freebase/20231108_nhdp_tree_freebase_po_25_1000.csv' 'data/fb15k-237/vocab_po_20231106161156.txt' 'output/tree/freebase/20231108_nhdp_tree_freebase_po_t25_b1_1000.csv'
python save_tree.py '../nHDP_matlab/output/tree/freebase/20231207-075638_nhdp_tree_freebase_po_t25_b01_1000.csv' 'data/fb15k-237/vocab_po_20231207-075638.txt' 'output/tree/freebase/20231207-075638_nhdp_tree_freebase_po_t25_b01_1000.csv'

# evaluation command
python evaluation/evaluation_scripts.py 'output/test/20240102_nhdp_tree_freebase_po_t22_b1_1000_172952.csv' 'data/fb15k-237/subject_documents.pkl' 'evaluation_score/freebase/20240102_nhdp_tree_fb_po.json'
python evaluation/evaluation_scripts.py 'output/tree/freebase/20240102_nhdp_tree_freebase_po_t22_b01_1000_172952.csv' 'data/fb15k-237/subject_documents_20240102172952.pkl' 'evaluation_score/freebase/20240102_nhdp_tree_fb_po_b01.json'
python evaluation/evaluation_scripts.py 'output/tree/freebase/20231108_nhdp_tree_freebase_po_t25_b1_1000.csv' 'data/fb15k-237/subject_documents_20231108-232426.pkl' 'evaluation_score/freebase/20231108_nhdp_tree_fb_po_b1.json'
python evaluation/evaluation_scripts.py 'output/tree/freebase/20231207-075638_nhdp_tree_freebase_po_t25_b01_1000.csv' 'data/fb15k-237/subject_documents_20231207-075638.pkl' 'evaluation_score/freebase/20231207-075638_nhdp_tree_fb_po_b01.json'

=====================2.dbpedia
nhdp_evaluation_dbpedia: preprocessing ready for nhdp
matlab nhdp
write_tree_csv(Tree, 'output/tree/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.csv')

# testing
python print-tree.py '../nHDP_matlab/output/tree/dbpedia/nhdp_tree_docs.csv' 'data/dbpedia/vocab_poseperate.txt'

python save_tree.py '../nHDP_matlab/output/tree/dbpedia/nhdp_tree_docs.csv' 'data/dbpedia/vocab_poseperate.txt' 'output/tree/dbpedia/20240409_nhdp_tree_dbpedia_po.csv'

python evaluation/evaluation_scripts.py 'output/tree/dbpedia/20240409_nhdp_tree_dbpedia_po.csv' 'data/dbpedia/subject_documents.pkl' 'evaluation_score/dbpedia/20240409_nhdp_tree_dbpedia_po.json'

# foraml evaluation
python print-tree.py '../nHDP_matlab/output/tree/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.csv' 'data/dbpedia/vocab_po_20240414.txt'

python save_tree.py '../nHDP_matlab/output/tree/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.csv' 'data/dbpedia/vocab_po_20240414.txt' 'output/tree/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.csv'

python evaluation/evaluation_scripts.py 'output/tree/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.csv' 'data/dbpedia/subject_documents_20240414.pkl' 'evaluation_score/dbpedia/20240414_nhdp_tree_dbpedia_po_t2231_b1_1000.json'


=====================3.wikidata
write_tree_csv(Tree, 'output/tree/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.csv')


python print-tree.py '../nHDP_matlab/output/tree/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.csv' 'data/wikidata5m_inductive/vocab_po_20240414_subset_train.txt'

python save_tree.py '../nHDP_matlab/output/tree/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.csv' 'data/wikidata5m_inductive/vocab_po_20240414_subset_train.txt' 'output/tree/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.csv'

python evaluation/evaluation_scripts.py 'output/tree/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.csv' 'data/wikidata5m_inductive/subject_documents_20240414_subset_train.pkl' 'evaluation_score/wikidata/20240414_nhdp_tree_wikidata_po_t23_b01_1000.json'

=====================3.wikidata
write_tree_csv(Tree, 'output/tree/webred/20240418_nhdp_tree_webred_po_t23_b01_1000.csv')


python print-tree.py '../nHDP_matlab/output/tree/webred/20240418_nhdp_tree_webred_po_t23_b01_1000.csv' 'data/WebRED/vocab_po_20240418_subset_train.txt'

python save_tree.py '../nHDP_matlab/output/tree/webred/20240418_nhdp_tree_webred_po_t23_b01_1000.csv' 'data/WebRED/vocab_po_20240418_subset_train.txt' 'output/tree/webred/20240418_nhdp_tree_webred_po_t23_b01_1000.csv'

python evaluation/evaluation_scripts.py 'output/tree/webred/20240418_nhdp_tree_webred_po_t23_b01_1000.csv' 'data/WebRED/subject_documents_20240418_subset_train.pkl' 'evaluation_score/webred/20240414_nhdp_tree_wikidata_po_t23_b01_1000.json'


=====================topmost traco
cp /home/yujia/TopMost/topmost/preprocessing/preprocessing_kg.py /home/yujia/miniconda3/envs/torch2-tune/lib/python3.10/site-packages/topmost/preprocessing/preprocessing_kg.py
***********************python