import sys

sys.path.extend([".."])

import argparse
import json

import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
#tf.config.experimental.set_memory_growth(gpus[0], True)
import tensorflow_addons as tfa

import glob
import os

# import trainer
# import metrics MacroF1Score, Accuracy, MacroF1ScoreBI, EntityF1
from polus.callbacks import ConsoleLogCallback, TimerCallback, LossSmoothCallback, ValidationDataCallback, SaveModelCallback, EarlyStop, WandBLogCallback
from polus.utils import set_random_seed, complex_json_serializer
from polus.data import CachedDataLoader, build_bert_embeddings, DataLoader
from polus.schedulers import warmup_scheduler
from polus.ner.metrics import MacroF1Score, Accuracy
from polus.training import ClassifierTrainer
from polus.models import load_model

import modelsv2

from utils import get_temp_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, tokseqconcat_generator, random_augmentation, ShufflerAugmenter, NoiseAugmenter
from losses import sum_cross_entropy, weighted_cross_entropy, sample_weighted_cross_entropy
from corpora import NLMChemCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BaseCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1, EntityF1DocAgreement
from config import ROOT

from transformers.optimization_tf import WarmUp, AdamWeightDecay

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument("-from_model", type=str, default=None, help="A json file with the configuration of the model")
    parser.add_argument("-base_lr", type=float, default=0.01, help="The base learning rate for the optimizer")
    parser.add_argument("-batch_size", type=int, default=64, help="Batch size that will be used during training")
    parser.add_argument("-epoch", type=int, default=30, help="Number of epochs during train")
    parser.add_argument("-rnd_seed", type=int, default=42, help="Number for the random seed")
    parser.add_argument("-gaussian_noise", type=float, default=None, help="Gausian noise")
    parser.add_argument("-random_augmentation", type=str, default=None, help="Mode to use random data augmentation")
    parser.add_argument("-label_smoothing", type=float, default=0, help= "Amount of label smoothing to be used during cross entropy")
    parser.add_argument("-use_dice_loss", action='store_true', help= "Flag that replaces the cross entropy loss by the dice_loss")
    parser.add_argument("-use_sample_weights", action='store_true', help= "Flag that enables the training with sample weights")
    parser.add_argument("-use_class_weights", action='store_true', help= "Flag that enables the training with class weights")
    parser.add_argument("-use_crf_mask", action='store_true', help="Flag that enables the masking of impossible transitions on the CRF")
    parser.add_argument("-use_fulltext", action='store_true', help="Flag to use the fulltext of the corpus where the passages are discarded")
    parser.add_argument("-train_datasets", nargs='+', default=[], type=str, help="Flag to add the training datasets")
    parser.add_argument("-test_datasets", nargs='+', default=[], type=str, help="Flag to add the test datasets")
    parser.add_argument("-train_w_test", action='store_true', help="Flag to train with the test data")
    parser.add_argument("-use_nlmchem_test_syn", action='store_true', help="Flag to train with the test data")
    parser.add_argument("-wandb", type=str, default=None, help="WandB project name")
    args = parser.parse_args()
    
    if args.use_nlmchem_test_syn:
        KEY = "train_dev_test-r0.5-300docs"#"train_dev_test-r0.5-450docs"
    else:
        KEY = "train_dev-r0.5-200docs"
        
    NLMCHEMSYN_PATH = os.path.join(ROOT, 'datasets', 'NLMChemSyn', f'NLMChemSyn-{KEY}', 'unique.json')
        
    class NLMChemSynCorpus(BaseCorpus):

        def __init__(self):
            super().__init__({KEY: NLMCHEMSYN_PATH},
                             ignore_non_contiguous_entities=False,
                             ignore_normalization_identifiers=False,
                             solve_overlapping_passages=False)
    
    dataset_mapping = {
        "NLMCHEM": NLMChemCorpus,
        "synNLMCHEM": NLMChemSynCorpus,
        "CDR": CDRCorpus,
        "CHEMDNER": CHEMDNERCorpus,
        "DrugProt": DrugProtFilteredCorpus,
        #"CRAFT": CRAFTCorpus,
        #"BioNLP11": BioNLP11IDCorpus,
        #"BioNLP13CG": BioNLP13CGCorpus,
        #"BioNLP13PC": BioNLP13PCCorpus,
    }
    
    assert len(args.train_datasets)>0
    
    all_datasets = set(args.train_datasets) | set(args.test_datasets)
    print("Found following train and test datasets", all_datasets)
    
    assert len(all_datasets - set(dataset_mapping.keys()))==0
    
    BERT_CHECKPOINT = PUBMEDBERT_FULL
    
    PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
    
    temp_file_for_training_cache = get_temp_file()
    print(f"Using file {temp_file_for_training_cache} to cache the dataset transformations")
    
    # set the same random seed for reproducible results
    set_random_seed(args.rnd_seed)
    
    if args.from_model is None:
        cfg = {
            "embeddings":{
                "type":"bert",
                "checkpoint":BERT_CHECKPOINT,
                "bert_layer_index": -3, 
            },
            "model":{
                "sequence_length":256,
                "output_classes": 4,
                "low": 128, 
                "high": 384,
                "activation": "mish",
                "dropout_p": 0.3,
                "hidden_space": 900,
                "gaussian_noise": args.gaussian_noise,
            }
        }

        if args.use_crf_mask:
            cfg["model"]["mask_impossible_transitions"] = tf.constant([[1,1,1,1],[1,1,1,0],[1,1,1,1],[1,1,1,1]], dtype=tf.float32)
        else:
            cfg["model"]["mask_impossible_transitions"] = None
        
        model = modelsv2.BERT_MLP_DROPOUT_CRF_V2(**cfg)
    else:
        model = load_model(args.from_model, external_module=modelsv2)

        cfg = model.savable_config
   
    
    get_bert_embeddings = build_bert_embeddings(**cfg)
            
    def heavy_batch_map_function(data):

        output = get_bert_embeddings(input_ids = data["input_ids"], 
                                     token_type_ids = data["token_type_ids"],
                                     attention_mask = data["attention_mask"])
        
        data["embeddings"] = output["last_hidden_state"]

        return data
    
    #corpora = [NLMChemCorpus(), CDRCorpus(), CHEMDNERCorpus(), CRAFTCorpus(), BioNLP11IDCorpus(), BioNLP13CGCorpus(), BioNLP13PCCorpus(), DrugProtFilteredCorpus()]
    
    #corpora = [NLMChemCorpus(), CDRCorpus(), CHEMDNERCorpus(), DrugProtFilteredCorpus()] 
    
    dataloaders = {dataset_name: {} for dataset_name in all_datasets}
    
    corpora = {dataset_name:dataset_mapping[dataset_name]() for dataset_name in all_datasets}
    
    check_point_name = short_checkpoint_names[cfg["embeddings"]["checkpoint"]]
    _index = cfg["embeddings"]["bert_layer_index"]
    ## Building the DataLoaders by using the
    for dataset_n in dataloaders.keys():
        for group, corpus in corpora[dataset_n]:
            if args.use_fulltext:
                gen = bertseq_center_generator(
                            tokseqconcat_generator(
                                document_generator(corpus),
                                tokenizer=Tokenizer(model_name=BERT_CHECKPOINT)
                            )
                        )

            else:
                gen = bertseq_center_generator(
                            tokseq_generator(
                                  passage_generator(
                                    document_generator(corpus)
                                  ),
                                tokenizer=Tokenizer(model_name=BERT_CHECKPOINT)
                            )
                        )



            dataloaders[dataset_n][group] = CachedDataLoader(gen, 
                                                       cache_chunk_size = 512,
                                                       show_progress = True,
                                                       cache_folder = os.path.join(PATH_CACHE, "dataloaders"),
                                                       cache_additional_identifier = f"index{_index}_{check_point_name}",
                                                       accelerated_map_f=heavy_batch_map_function)
    
    print("Clean up any heavy model that is on the GPU")
    tf.keras.backend.clear_session()
    
    train_dls = []

    for dataset_n in args.train_datasets:
        if "train" in dataloaders[dataset_n]:
            train_dls.append(dataloaders[dataset_n]["train"])
            train_dls.append(dataloaders[dataset_n]["dev"])
            if args.train_w_test and "test" in dataloaders[dataset_n]:
                train_dls.append(dataloaders[dataset_n]["test"])
        else:
            # NLMChemSyn
            train_dls.append(dataloaders[dataset_n][KEY])
            
    #prepare data to be logged
    additional_info = {}
    for i, dl in enumerate(train_dls):
        additional_info[f"Train_CachedDataLoader_{i}"] = dl.cache_index_path
        print(f"Train_CachedDataLoader_{i}", dl.cache_index_path)
    
    
    
    validation_callbacks = []
    
    def custom_inference_f(model, sample):
        # describes more complex behaviour for the validator callback
        _sample = {
            "corpus": sample["corpus"],
            "group": sample["group"],
            "identifier": sample["identifier"],
            "spans": sample["spans"][:,cfg["model"]["low"]:cfg["model"]["high"]],
            "is_prediction": sample["is_prediction"][:,cfg["model"]["low"]:cfg["model"]["high"]],
            "tags_int": tf.cast(sample["tags_int"][:,cfg["model"]["low"]:cfg["model"]["high"]], tf.int32),
        }
        
        _sample["tags_int_pred"] = model.inference(sample["embeddings"], sample["attention_mask"])
        return _sample
    
    for dataset_n in args.test_datasets: 
        dl = dataloaders[dataset_n]["test"]
        additional_info[f"Validation_CachedDataLoader_{dataset_n}"] = dl.cache_index_path
        
        test_dls= dl.batch(args.batch_size)\
                    .prefetch(tf.data.experimental.AUTOTUNE)
        
        validation_callbacks.append(ValidationDataCallback(test_dls, name=f"{dataset_n}_Test", custom_inference_f=custom_inference_f))
        validation_callbacks.append(SaveModelCallback("best", validation_name=f"{dataset_n}_Test", metric_name="EntityF1", cache_folder=os.path.join(PATH_CACHE, "saved_models")))
    
    if len(train_dls) > 1:
        print("Merge the training dataloaders")
        train_dataloader = CachedDataLoader.merge(*train_dls)
    else:
        train_dataloader = train_dls[0]
    
    #def training_map(data):
    #    return {
    #        "embeddings": data["embeddings"], 
    #        "attention_mask":data["attention_mask"], 
    #        "tags_int":data["tags_int"]
    #    }
    #.map(training_map, num_parallel_calls=tf.data.AUTOTUNE)\
    if args.random_augmentation is not None:
        n_samples = 7800#train_dataloader.get_n_samples()
        cfg["augmentation"] = {
                "k_selections": 4,
                "prob_change_entity": 0.66,
                "prob_change_non_entity": 0.33,
            }
        if args.random_augmentation == "shuffle":
            
            augmenter = ShufflerAugmenter(k_selections = cfg["augmentation"]["k_selections"],
                                          prob_change_entity = cfg["augmentation"]["prob_change_entity"],
                                          prob_change_non_entity = cfg["augmentation"]["prob_change_non_entity"])
            
        elif args.random_augmentation == "noise":

            augmenter = NoiseAugmenter(k_selections = cfg["augmentation"]["k_selections"],
                                       prob_change_entity = cfg["augmentation"]["prob_change_entity"],
                                       prob_change_non_entity = cfg["augmentation"]["prob_change_non_entity"])
            
        else:
            raise ValueError(f"random_augmention received a non supported mode: {args.random_augmentation}")
        
        
        training_gen = random_augmentation(train_dataloader.pre_shuffle(),
                                           augmenter,
                                           samples_per_step=n_samples,
                                           shuffle=True)
        
        training_ds = DataLoader(training_gen).batch(args.batch_size, drop_remainder=True)\
                                              .prefetch(tf.data.AUTOTUNE)
        
    else:
        n_samples = train_dataloader.get_n_samples()
        
        training_ds = train_dataloader.pre_shuffle()\
                                      .shuffle(min(30000, n_samples))\
                                      .batch(args.batch_size, drop_remainder=True)\
                                      .prefetch(tf.data.AUTOTUNE)
    #def testing_map(data):
    #    data["embeddings"] = data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:]
    #    data["spans"] = data["spans"][cfg["model"]["low"]:cfg["model"]["high"]]
    #    data["is_prediction"] = data["is_prediction"][cfg["model"]["low"]:cfg["model"]["high"]]
    #    data["tags_int"] = tf.cast(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], tf.int32)
    #    return data
        
    # init the model
    if args.from_model is None:
        _sample = next(iter(training_ds))
        model.init_from_data(**{"embeddings": _sample["embeddings"], "attention_mask": _sample["attention_mask"]})
    model.summary()
    
    
        
    negative_weight = 0.4
    
    #prepare data to be logged
    additional_info["negative_weight"] = negative_weight
    
    if hasattr(model, 'loss'):
        if args.use_sample_weights:
            loss = model.loss_sample_weights([0, 0, 1, 1], negative_weight)
        elif args.use_class_weights:
            loss = model.loss_class_weights
        else:
            loss = model.loss
    else:

        if args.use_sample_weights:
            loss = sample_weighted_cross_entropy()
        elif args.use_class_weights:
            loss = weighted_cross_entropy()
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing) 
    
    epoch = args.epoch
    
    steps =  n_samples//args.batch_size
    """
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay([steps, steps*(epoch//3), steps*(epoch*2//3)], [1e-0, 1e-1, 1e-2, 1e-3])
    # lr and wd can be a function or a tensor
    lr = args.base_lr * schedule(step)
    wd = lambda: 5e-4 * schedule(step)

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    
    """
    optimizer = AdamWeightDecay(
        learning_rate = warmup_scheduler((steps+1)*epoch, args.base_lr),
        weight_decay_rate = 1e-4,
        exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"],
    )
    
    
    def get_ytrue_and_ypred(sample):
        return sample["tags_int"], sample["tags_int_pred"]
    
    test_corpora = [corpora[dataset_name] for dataset_name in args.test_datasets]
    
    trainer = ClassifierTrainer(model,
                            optimizer,
                            loss,
                            metrics=[MacroF1Score(num_classes=4, reduce_f = get_ytrue_and_ypred),
                                     EntityF1(test_corpora),
                                     EntityF1DocAgreement(test_corpora),
                                     Accuracy(num_classes=4, reduce_f = get_ytrue_and_ypred)])

    
    
    callbacks = [LossSmoothCallback(output=True), # if output is True the smooth should be positioned before all the streaming outputs
                 TimerCallback()] # This callback should be positioned before all the streaming outputs
    
    callbacks += validation_callbacks
    
    if args.wandb is not None:
        callbacks += [WandBLogCallback(args.wandb, args, entity='bitua', additional_info = additional_info)]
    
    callbacks += [SaveModelCallback("end", cache_folder=os.path.join(PATH_CACHE, "saved_models")),
                 ConsoleLogCallback(), # Prints the training on the console
                 EarlyStop(),
                ]
    
    
    def train_transformation(data):
        x = {"embeddings":data["embeddings"], "attention_mask":data["attention_mask"]}
        y = tf.one_hot(data["tags_int"][:,cfg["model"]["low"]:cfg["model"]["high"]], cfg["model"]["output_classes"])
        return x, y
    
    
    print()
    print("Args")
    for arg in vars(args):
        print("\t",arg, ": ",getattr(args, arg))
    print()
    print(json.dumps(complex_json_serializer(cfg), indent=4, sort_keys=True))
    
    trainer.train(tf_dataset=training_ds, 
                  epochs=epoch,
                  steps = steps,
                  train_map_f = train_transformation,
                  callbacks=callbacks)
    

    
    