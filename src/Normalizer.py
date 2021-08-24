from corpora import NLMChemCorpus#, CDRCorpus, CHEMDNERCorpus
from NormalizerUtils import dictionaryLoader, mapWithoutAb3P, mapWithAb3P
from elements import merge_collections

import json



class Normalizer():
	def normalize(annotations, goldStandard, test=False):

		test = True

		ab3pAbbreviationExpansion = True
		ab3pExpansionDictLevels = ["Document", "Corpus"]
		ab3pDictLevel = ab3pExpansionDictLevels[1]
		meshDictionaries = ["MeSH_Dxx","SCR"]
		outputForIndexingFilename = "outputForBaldGuy.json"
		outputCorpusFilename = "nlm_chem_test_bioc.json"

		trainNLMCorpus = NLMChemCorpus()
		trainCollection = trainNLMCorpus["train"]
		devCollection   = trainNLMCorpus["dev"]
		testCollection  = trainNLMCorpus["test"]


		meshDictionary = dictionaryLoader(meshDictionaries)

		if not test:
			trainCorpus = merge_collections(trainCollection, devCollection)
			testCorpus  = testCollection

			# trainCorpus, mappedDocuments = mapWithoutAb3P(trainCorpus)
			trainCorpus, mappedDocuments = mapWithAb3P(trainCorpus, meshDictionary, ab3pDictLevel)

		else:
			# trainCorpus = merge_collections(trainCollection, devCollection, testCollection)
			testCorpus = testCollection
			testCorpus, mappedDocuments = mapWithAb3P(testCorpus, meshDictionary, ab3pDictLevel)



		with open(outputForIndexingFilename, "w") as file:
			json.dump(mappedDocuments, file, indent=4)


		with open(outputCorpusFilename, "w") as file:
			newJson = testCorpus.pretty_json()
			_ = file.write(newJson)


		return []

#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ./nlm_chem_train_bioc.json --evaluation_type identifier --evaluation_method strict --annotation_type Chemical
