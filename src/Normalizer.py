from corpora import NLMChemCorpus, NLMChemTestCorpus #, CDRCorpus, CHEMDNERCorpus
from NormalizerUtils import dictionaryLoader, mapWithoutAb3P, mapWithoutAb3P_AugmentedDictionary, mapWithAb3P, mapWithAb3P_AugmentedDictionary
from elements import merge_collections

import json



class Normalizer():
	def normalize(annotations, goldStandard, test=False):
		if test:
			print("\nRunning the normalization pipeline on the test corpus.\n")

#
#
# SETTING IMPORTANT VARIABLES TO DECIDE WHAT PIPELINE SHOULD BE USED
#
#
		dictionaryDatasetAugmentation = True
		ab3pAbbreviationExpansion = True

		ab3pExpansionDictLevels = ["Document", "Corpus"]
		ab3pDictLevel = ab3pExpansionDictLevels[1]

		meshDictionaries = ["MeSH_Dxx","SCR"]

		outputForIndexingFilename = "outputForBaldGuy.json"
		outputCorpusFilename = "nlm_chem_test_bioc.json"


#
#
# FINISHED SETTING VARIABLES
#
#

		trainNLMCorpus = NLMChemCorpus()
		trainCollection = trainNLMCorpus["train"]
		devCollection   = trainNLMCorpus["dev"]
		testCollection  = trainNLMCorpus["test"]

		testNLMCorpus = NLMChemTestCorpus()
		trueTestCollection = testNLMCorpus["test"]


		meshDictionary = dictionaryLoader(meshDictionaries)

		if not test:
			trainCorpus = merge_collections(trainCollection, devCollection)
			testCorpus  = testCollection
		else:
			trainCorpus = merge_collections(trainCollection, devCollection, testCollection)
			testCorpus  = trueTestCollection


		if ab3pAbbreviationExpansion:
			if dictionaryDatasetAugmentation:
				_, _, meshDictionary, abbreviationMap = mapWithAb3P_AugmentedDictionary(trainCorpus, meshDictionary, ab3pDictLevel, test=False)
			else:
				_, _, abbreviationMap = mapWithAb3P(trainCorpus, meshDictionary, ab3pDictLevel)
			testCorpus, mappedDocuments, _ = mapWithAb3P(testCorpus, meshDictionary, ab3pDictLevel, abbreviationMap)

		else:
			if dictionaryDatasetAugmentation:
				_, _, meshDictionary = mapWithoutAb3P_AugmentedDictionary(trainCorpus, meshDictionary, test=False)
			testCorpus, mappedDocuments = mapWithoutAb3P(testCorpus, meshDictionary)



		with open(outputForIndexingFilename, "w") as file:
			json.dump(mappedDocuments, file, indent=4)


		with open(outputCorpusFilename, "w") as file:
			newJson = testCorpus.pretty_json()
			_ = file.write(newJson)


		return []

## DEVELOPMENT EVALUATION
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ./nlm_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type Chemical
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-test.BioC.json --prediction_path ./nlm_chem_test_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type Chemical

## TRUE TEST EVALUATION
#python3 ./scripts-rui/filter_run_for_chemical_identification_evaluation_on_final_test.py nlm_chem_test_bioc.json
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/test/BC7T2-NLMChemTest-annotated_v1.BioC.json --prediction_path ./nlm_chem_test_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type Chemical

## Run normalization in test corpus
# python3 main.py -n -t