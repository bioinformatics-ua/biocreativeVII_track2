import argparse
import configparser
#from Utils import Utils
from UtilsThatWork import Utils
from Annotator import Annotator
from Normalizer import Normalizer
from Indexer import Indexer

def help(show=False):
	parser = argparse.ArgumentParser(description="")
	configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
	configs.add_argument('-s', '--settings', dest='settings', \
						type=str, default="settings.ini", \
						help='The system settings file (default: settings.ini)')	
	configs.add_argument('-a', '--annotate', default=False, action='store_true', \
						 help='Flag to annotate the files (default: False)')
	configs.add_argument('-n', '--normalize', default=False, action='store_true', \
							help='Flag to normalize the detected concepts (default: False)')
	configs.add_argument('-i', '--indexing', default=False, action='store_true', \
							help='Flag to index the detected concepts (default: False)')
	configs.add_argument('-m', '--merging', default=False, action='store_true', \
							help='Flag to merge datasets (default: False)')

	executionMode = parser.add_argument_group('Execution Mode', 'Flags to select the execution mode!')
	#executionMode.add_argument('-tr', '--train', default=False, action='store_true', \
	#						help='In this mode, the script will work to train the models (default: False)')
	executionMode.add_argument('-t', '--test', default=False, action='store_true', \
							help='In this mode, the script will work using the test dataset(default: False)')
	executionMode.add_argument('-tt', '--ttest', default=False, action='store_true', \
							help='In this mode, the script will work using test partition from training dataset(default: False)')
	executionMode.add_argument('-st', '--study', default=False, action='store_true', \
							help='In this mode, the script will change the rules weights (default: False)')
	
	if show:
		parser.print_help()
	return parser.parse_args()

def readSettings(settingsFile):
	configuration = configparser.ConfigParser()
	configuration.read(settingsFile)
	if not configuration:
		raise Exception("The settings file was not found!")
	return configuration

def main():
	args = help()
	settings = readSettings(args.settings)
	if  not args.annotate and \
		not args.normalize and \
		not args.indexing and \
		not args.merging:
		print("Nothing to do, please type --help to show the different options!")
		help(show=True)
		exit()

	files, goldStandard = Utils.readFiles()
	# gsAnn = goldStandard["annotations"] if not args.test else False
	# gsIndexing = goldStandard["indexing"] if not args.test else False
	gsAnn = []
	gsIndexing = []
	meshList = None

	if args.annotate:
		annotations = Annotator.annotate(files, goldStandard=gsAnn, test=args.test)

	if args.normalize:
		if not args.annotate:
			annotations = Utils.readAnnotations()
		meshList = Normalizer.normalize(annotations, goldStandard=gsAnn, test=args.test)
		Utils.buildIndentificationSubmission(meshList)

	if args.indexing:
		readMeshList = Utils.readMesh(settings["files"]["mesh"])
		Indexer.index(mesh=readMeshList, train_test=args.ttest, test=args.test, study=args.study)

	if args.merging:
		Utils.merge(settings["datasets"]["train"])

	print("Done!")
main()