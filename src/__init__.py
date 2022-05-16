import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('source_directory',
                        type=str,
                        help='')
    
    configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
    configs.add_argument('-s', '--settings', dest='settings', \
                        type=str, default="src/settings.yaml", \
                        help='The system settings file (default: settings.yaml)')
    configs.add_argument('-a', default=False, action='store_true', \
                         help='Flag to annotate the files (default: False)')
    configs.add_argument('-n', default=False, action='store_true', \
                            help='Flag to normalize the detected concepts (default: False)')
    configs.add_argument('-i', default=False, action='store_true', \
                            help='Flag to index the detected concepts (default: False)')
    
    #subparser_annotator = parser.add_subparsers(dest="action")
    #annotator = subparser_annotator.add_parser('annotator')
    #subparser_normalizer = parser.add_subparsers(dest="action")
    #normalizer = subparser_normalizer.add_parser('normalizer')
    #subparser_indexer = parser.add_subparsers(dest="action")
    #indexer = subparser_indexer.add_parser('indexer')
    
    
    indexer_configs = parser.add_argument_group('Indexer settings', 'This settings are related to the indexer module.')
    indexer_configs.add_argument('--indexer.write_path', dest='indexer_write_path', \
                                 type=str, default="outputs/indexer", \
                                 help='The indexer outputs path')
    
    args = parser.parse_args()
    #print()