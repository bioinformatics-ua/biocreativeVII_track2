from corpora import NLMChemCorpus, NLMChemTestCorpus#, CDRCorpus, CHEMDNERCorpus

PERCENTAGE = 1 #aumentar, requer mais conceitos anotados
MIN_OCCUR_CAPTIONS = 0.16 * PERCENTAGE
MIN_OCCUR_ABSTRACT = 0.17 * PERCENTAGE
MIN_OCCUR_TITLE = 0.06 * PERCENTAGE #best 0.06
MIN_OCCUR_CONCL = 0.06 * PERCENTAGE #best 0.06
METHOD = 1

#evaluator
#python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-train.BioC.json --prediction_path ../results/train.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

#dev
#python3 main.py -i && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-dev.BioC.json --prediction_path ../results/train.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical

#test
#python3 main.py -i && python3 ./evaluation/evaluate.py --reference_path ../dataset/NLM-CHEM/train/BC7T2-NLMChem-corpus-test.BioC.json --prediction_path ../results/train.json --evaluation_type identifier --evaluation_method strict --annotation_type MeSH_Indexing_Chemical


class Indexer():
	def index(mesh, test=False):
		fileName = "train.json"
		if True: #test:
			corpus = NLMChemTestCorpus()
			corpus = corpus["test"]
			fileName = "Track2-Team-110-Subtask2-Indexing-Run-{}.json".format(4)
		else:
			corpus = NLMChemCorpus()
			corpus = corpus["train"]
			print(corpus)
			mesh = Indexer.readGoldStandard(corpus)

		corpusWithScores = Indexer.getScores(corpus, mesh, Indexer.getDicts(corpus))

		with open("../results/{}".format(fileName), "w") as file:
			newJson = corpusWithScores.pretty_json()
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



