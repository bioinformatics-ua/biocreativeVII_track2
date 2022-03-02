from corpora import NLMChemCorpus, NLMChemTestCorpus#, CDRCorpus, CHEMDNERCorpus
from elements import merge_collections

PERCENTAGE = 1 #aumentar, requer mais conceitos anotados
MIN_OCCUR_CAPTIONS = 1000 * PERCENTAGE
MIN_OCCUR_ABSTRACT = 1000 * PERCENTAGE
MIN_OCCUR_TITLE = 1000 * PERCENTAGE #best 0.06
MIN_OCCUR_CONCL = 1000 * PERCENTAGE #best 0.06


#pos desafio
#MIN_OCCUR_CAPTIONS = 0.16 * PERCENTAGE
#MIN_OCCUR_ABSTRACT = 0.17 * PERCENTAGE
#MIN_OCCUR_TITLE = 0.06 * PERCENTAGE
#MIN_OCCUR_CONCL = 0.06 * PERCENTAGE

#submission
#MIN_OCCUR_CAPTIONS = 0.2 * PERCENTAGE
#MIN_OCCUR_ABSTRACT = 0.07 * PERCENTAGE
#MIN_OCCUR_TITLE = 0.1 * PERCENTAGE
#MIN_OCCUR_CONCL = 0.06 * PERCENTAGE

#melhores maximos isolados
#MIN_OCCUR_CAPTIONS = 0.06 * PERCENTAGE
#MIN_OCCUR_ABSTRACT = 0.06 * PERCENTAGE
#MIN_OCCUR_TITLE = 0.01 * PERCENTAGE
#MIN_OCCUR_CONCL = 0.01 * PERCENTAGE

#melhores conjunto 3
MIN_OCCUR_CAPTIONS = 0.19 * PERCENTAGE
MIN_OCCUR_ABSTRACT = 0.15 * PERCENTAGE
MIN_OCCUR_TITLE = 0.02 * PERCENTAGE
MIN_OCCUR_CONCL = 0.1 * PERCENTAGE

#melhor bm
MIN_OCCUR_CAPTIONS = 0.22 * PERCENTAGE
MIN_OCCUR_ABSTRACT = 0.1 * PERCENTAGE
MIN_OCCUR_TITLE = 0.02 * PERCENTAGE
MIN_OCCUR_CONCL = 0.1 * PERCENTAGE


METHOD = 1

#evaluator for dataset train
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ../results/train.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

#train_dev
#python3 main.py -i && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train-dev.BioC.json --prediction_path ../results/train_dev.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical


#train
#python3 main.py -i && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ../results/train_train.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

#dev
#python3 main.py -i && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-dev.BioC.json --prediction_path ../results/train_dev.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

#test
#python3 main.py -i -tt && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-test.BioC.json --prediction_path ../results/train_test.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical


#evaluator for dataset test
#python3 main.py -i -t && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/test/BC7T2-NLMChemTest-indexed_v1.BioC.json --prediction_path ../results/test.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical



class Indexer():
	def index(mesh, train_test=None, test=None, study=None):
		if test: #test:
			corpus = NLMChemTestCorpus()
			corpus = corpus["test"]
			fileName = "test.json"
			#mesh are read from annotations (JS provide me)
		elif train_test:
			corpus 		= NLMChemCorpus()
			corpus 		= corpus["test"]
			fileName 	= "train_test.json"
			mesh 		= Indexer.readGoldStandard(corpus)
		else:
			trainNLMCorpus 	= NLMChemCorpus()
			trainCollection = trainNLMCorpus["train"]
			devCollection   = trainNLMCorpus["dev"]
			corpus 			= merge_collections(trainCollection, devCollection)
			mesh 			= Indexer.readGoldStandard(corpus)
			fileName 		= "train_dev.json"
			
			if study:
				for x in range(0, 31):
					global MIN_OCCUR_ABSTRACT
					MIN_OCCUR_ABSTRACT 	= x /100
					for y in range(0, 31):
						global MIN_OCCUR_CAPTIONS
						MIN_OCCUR_CAPTIONS 	= y /100
						for z in range(0, 31):
							global MIN_OCCUR_TITLE
							MIN_OCCUR_TITLE 	= z /100
							fileName 			= "train_dev_{}-{}-{}.json".format(x, y, z)
							print(fileName)
							Indexer.process(fileName, corpus, mesh)
				return None

		Indexer.process(fileName, corpus, mesh)

	def process(fileName, corpus, mesh):
		corpusIndexed = Indexer.getScores(corpus, mesh, Indexer.getDicts(corpus))
		with open("../results/{}".format(fileName), "w") as file:
			newJson = corpusIndexed.pretty_json()
			_ = file.write(newJson)

	def getScores(corpus, meshes, dicts):
		sections = dicts["sections"]
		captions = dicts["captions"]
		abstracts = dicts["abstracts"]
		conclusions = dicts["conclusions"]

		countMeshes = {}
		countAllMeshesByDoc = {}
		for id in meshes:
			countMeshes[id] = {}
			countAllMeshesByDoc[id] = 0
			for x in meshes[id]:
				mesh = x[0][0]
				if mesh == "-":
					continue
				if mesh not in countMeshes[id]:
					countMeshes[id][mesh] = 0
				countMeshes[id][mesh] += 1
				countAllMeshesByDoc[id] += 1

		indexedMeshsByDoc = {}
		for id in meshes:
			meshesToAdd = set()
			for x in meshes[id]:
				mesh = x[0][0]
				span = x[1]
				spanStart = span[0]
				spanEnd = span[1]
				if mesh == "-":
					continue

				if METHOD == 1:
					#Check if mesh is a section title
					for sec in sections[id]:
						if spanStart > sections[id][sec][0] and spanEnd < sections[id][sec][1]:
							#mesh in title
							if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_TITLE:
								meshesToAdd.add(mesh)

					#Check if mesh is in a caption
					for sec in captions[id]:
						if spanStart > captions[id][sec][0] and spanEnd < captions[id][sec][1]:
							#mesh in caption
							if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_CAPTIONS:
								meshesToAdd.add(mesh)

					#Check if is in abstract
					for sec in abstracts[id]:
						if spanStart > abstracts[id][sec][0] and spanEnd < abstracts[id][sec][1]:
							if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_ABSTRACT:
								meshesToAdd.add(mesh)

					#Check if is in abstract
					for sec in conclusions[id]:
						if spanStart > conclusions[id][sec][0] and spanEnd < conclusions[id][sec][1]:
							if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_CONCL:
								meshesToAdd.add(mesh)

				elif METHOD == 2:
					for sec in sections[id]:
						if spanStart > sections[id][sec][0] and spanEnd < sections[id][sec][1]:
							#mesh in title
							pass#meshesToAdd.add(mesh)
					
					#Check if is in abstract
					for sec in abstracts[id]:
						if spanStart > abstracts[id][sec][0] and spanEnd < abstracts[id][sec][1]:
							meshesToAdd.add(mesh)
							print(mesh)
							
					'''
					#Check if mesh is in a caption
					for sec in captions[id]:
						if spanStart > captions[id][sec][0] and spanEnd < captions[id][sec][1]:
							#mesh in caption
							if countMeshes[id][mesh]/countAllMeshesByDoc[id] > MIN_OCCUR_CAPTIONS:
								if mesh in abstractMeshes:
									meshesToAdd.add(mesh)
					'''
				else:
					print("NO METHOD DEFINED!!!")

			indexedMeshsByDoc[id] = list(meshesToAdd)
			corpus[id].set_mesh_indexing_identifiers(list(meshesToAdd))
		Indexer.calculateMetrics(indexedMeshsByDoc)
		return corpus

	def calculateMetrics(indexedMeshsByDoc):
		minimum = 10000
		maximum = -1
		avg = 0
		countWithoutMesh = 0
		for meshes in indexedMeshsByDoc:
			count = len(indexedMeshsByDoc[meshes])
			if count == 0:
				countWithoutMesh += 1

			if count < minimum:
				minimum = count

			if count > maximum:
				maximum = count

			avg += count
		avg /= len(indexedMeshsByDoc)
		print("Without Mesh: {}\nMinimum: {}\nMaximum: {}\nAverage: {}\n\n".format(countWithoutMesh, minimum, maximum, avg))


	def getDicts(corpus):
		dicts = {}
		dicts["sections"] = {} #docID:[section:startSpan]
		dicts["captions"] = {} #docID:[captions:span tuple]
		dicts["abstracts"] = {}#docID:[captions:span tuple]
		dicts["conclusions"] = {}#docID:[captions:span tuple]

		for id, document in corpus:
			dicts["sections"][id] = {}
			dicts["captions"][id] = {}
			dicts["abstracts"][id] = {}
			dicts["conclusions"][id] = {}

			for passage in document:
				if "fig_" in passage.typ or "table_" in passage.typ:
					dicts["captions"][id][passage.text] = passage.span

				elif "title" in passage.typ or "front" in passage.typ:
					dicts["sections"][id][passage.text] = passage.span

				if "abstract" in passage.typ:
					dicts["abstracts"][id][passage.text] = passage.span

				if "CONCL" in passage.section_type:
					dicts["conclusions"][id][passage.text] = passage.span

		return dicts

	def readGoldStandard(corpus):
		meshes = {}
		for id, document in corpus:
			meshes[id] = []
			for passage in document:
				for ide in passage.nes:
					for mesh in ide.identifiers:
						meshes[id].append([[mesh],ide.span])
		return meshes



