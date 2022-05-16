import os
import random
import string
import glob
import sys

def get_temp_file(path="/backup/biocreative_extension_track2/temp_files/"):
    import atexit
    import signal
    
    file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    file = os.path.join(path, file_name)
    
    def remove_files(*args):
        
        # delete the temp file
        for f in glob.glob(f"{file}*"):
            print("Deleting:",f)
            os.remove(f)
        
        sys.exit(0)
    
    # code to execute when sigterm and sigint :) 
    signal.signal(signal.SIGINT, remove_files)
    signal.signal(signal.SIGTERM, remove_files)
    
    # at the end of the program
    atexit.register(remove_files)
    
    return file


def write_collections_to_file(collections, path=None, suffix="", name=None):
    
    files_names = []
    
    for corpus in collections:
        for group in collections[corpus]:
            if group.startswith("BaseCorpus"):
                _name = f"{group}{suffix}.json"
            else:
                _name = f"{corpus}_{group}{suffix}.json"
            
            if name is not None:
                _name = os.path.join(name, _name)
            
            if path is not None:
                _name = os.path.join(path, _name)
            
            with open(_name, "w") as f:
                print(f"writing {corpus} {group} to {_name}")
                files_names.append(_name)
                f.write(collections[corpus][group].pretty_json())
                
    return files_names
    