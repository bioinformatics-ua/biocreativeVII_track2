from core import IModule

class Annotator(IModule):
    
    def __init__(self, 
                 write_output,
                 write_path):
        super().__init__()
        self.write_output = write_output
        self.write_path = write_path
    
    def transform(self, inputs):
        print("run annotor with",inputs)
        return ["from annotator"]
