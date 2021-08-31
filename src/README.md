## 

### Evaluation commands

#### Train

python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ./nlm_index_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

TP = 99, FP = 134, FN = 105
P = 0.4249, R = 0.4853, F = 0.4531

#### Test

python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-test.BioC.json --prediction_path ./nlm_index_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

TP = 48, FP = 67, FN = 61
P = 0.4174, R = 0.4404, F = 0.4286

#### Dev 

python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-dev.BioC.json --prediction_path ./nlm_index_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

TP = 30, FP = 30, FN = 21
P = 0.5000, R = 0.5882, F = 0.5405