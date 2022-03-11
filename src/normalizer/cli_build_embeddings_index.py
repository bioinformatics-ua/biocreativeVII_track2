import os

os.environ["POLUS_JIT"]="false"
import argparse
import pickle

from normalizer.embeddings import build_normalized_index, build_mesh_embedding_encoder

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Script for building the embedding index')
    parser.add_argument("mesh_file", type=str, nargs='+', help="A json file with the configuration of the model")
    
    args = parser.parse_args()
        
    get_mesh_embedding = build_mesh_embedding_encoder()
    
    print("Building mesh dictionary from the files:",  args.mesh_file)
    index_mesh = mesh_from_file(*args.mesh_file)
    
    shorter_names = [os.path.basename(path).split(".")[0] for path in args.mesh_file]
    
    cache_name = "_".join(shorter_names)
    
    print("Build norm matrix")
    normalized_mesh_index, normalized_emb_matrix = build_normalized_index(index_mesh, get_mesh_embedding, normalize_vectors=True)

    with open(os.path.join(f"{cache_name}.matrix"), "wb") as f:
        pickle.dump(normalized_emb_matrix, f)

    with open(os.path.join(f"{cache_name}.mindex"), "wb") as f:
        pickle.dump(normalized_mesh_index, f)
    
