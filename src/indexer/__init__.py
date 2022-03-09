from core import IModule

class Indexer(IModule):
    
    def __init__(self, 
                 write_output,
                 write_path):
        super().__init__()
        self.write_output = write_output
        self.write_path = write_path
    
    def transform(self, inputs):
        print("run indexer with",inputs)
        return ["from indexer"]
