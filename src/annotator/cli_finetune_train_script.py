import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa

import glob
import os

# import trainer
# import metrics MacroF1Score, Accuracy, MacroF1ScoreBI, EntityF1
from polus.callbacks import ConsoleLogCallback, TimerCallback, LossSmoothCallback, ValidationDataCallback, SaveModelCallback, EarlyStop, WandBLogCallback
from polus.utils import set_random_seed
from polus.data import CachedDataLoader, build_bert_embeddings
from polus.schedulers import warmup_scheduler
from polus.ner.metrics import MacroF1Score, Accuracy
from polus.training import ClassifierTrainer
from polus.models import load_model

import modelsv2

from utils import get_temp_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, SequenceDecoder
from losses import sum_cross_entropy, weighted_cross_entropy, sample_weighted_cross_entropy
from corpora import NLMChemCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BC5CDRCorpus, CRAFTCorpus, BioNLP11IDCorpus, BioNLP13CGCorpus, BioNLP13PCCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1, EntityReconstructedF1

from transformers.optimization_tf import WarmUp, AdamWeightDecay

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument("model_path", type=str, help="the path for the pretrained model")
    parser.add_argument("-base_lr", type=float, default=0.01, help="The base learning rate for the optimizer")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size that will be used during training")
    parser.add_argument("-epoch", type=int, default=30, help="Number of epochs during train")
    parser.add_argument("-label_smoothing", type=float, default=0, help= "Amount of label smoothing to be used during cross entropy")
    parser.add_argument("-use_dice_loss", action='store_true', help= "Flag that replaces the cross entropy loss by the dice_loss")
    parser.add_argument("-use_sample_weights", action='store_true', help= "Flag that enables the training with sample weights")
    parser.add_argument("-use_class_weights", action='store_true', help= "Flag that enables the training with class weights")
    parser.add_argument("-use_crf_mask", action='store_true', help="Flag that enables the masking of impossible transitions on the CRF")
    
    args = parser.parse_args()
    
    BERT_CHECKPOINT = PUBMEDBERT_FULL
    
    PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
    
    temp_file_for_training_cache = get_temp_file()
    print(f"Using file {temp_file_for_training_cache} to cache the dataset transformations")
    
    # set the same random seed for reproducible results
    set_random_seed()
    # needed for the evaluation metrics
    corpora = [NLMChemCorpus()]
    
    model = load_model(args.model_path, external_module=modelsv2)
    model.summary()
    
    cfg = model.savable_config
    
    dataloaders = {"train":[], "dev":[], "test":[]}
    
    check_point_name = short_checkpoint_names[cfg["embeddings"]["checkpoint"]]
    _index = cfg["embeddings"]["bert_layer_index"]
    ## Building the DataLoaders by using the 
    for group in dataloaders.keys():
        for corpus in corpora:
            if group in corpus:
                dataloaders[group].append(CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "dataloaders", f"index{_index}_{check_point_name}_heavy_batch_map_function_{corpus}_{group}_document_passage_tokseq_bertseq_center.index")))

    train_dls = dataloaders["train"]+dataloaders["dev"]#+dataloaders["test"][1:]
    test_dls = [dataloaders["test"][0]]
    
    print("Merge the training dataloaders")
    train_dataloader = CachedDataLoader.merge(*train_dls)
    
    n_samples = train_dataloader.get_n_samples()
    
    #def training_map(data):
    #    return {
    #        "embeddings": data["embeddings"], 
    #        "attention_mask":data["attention_mask"], 
    #        "tags_int":data["tags_int"]
    #    }
    #.map(training_map, num_parallel_calls=tf.data.AUTOTUNE)\
 
    training_ds = train_dataloader.pre_shuffle()\
                                  .shuffle(30000)\
                                  .batch(args.batch_size, drop_remainder=True)\
                                  .prefetch(tf.data.AUTOTUNE)
    #def testing_map(data):
    #    data["embeddings"] = data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:]
    #    data["spans"] = data["spans"][cfg["model"]["low"]:cfg["model"]["high"]]
    #    data["is_prediction"] = data["is_prediction"][cfg["model"]["low"]:cfg["model"]["high"]]
    #    data["tags_int"] = tf.cast(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], tf.int32)
    #    return data

    
    test_ds_NLMCHEM = test_dls[0].batch(args.batch_size*2)\
                                 .prefetch(tf.data.experimental.AUTOTUNE)
        
    #prepare data to be logged
    additional_info = {}
    for i, dl in enumerate(train_dls):
        additional_info[f"Train_CachedDataLoader_{group}_{i}"] = dl.cache_index_path
    
    for i, dl in enumerate(test_dls):
        additional_info[f"Validation_CachedDataLoader_{group}_{i}"] = dl.cache_index_path
        
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
    
    trainer = ClassifierTrainer(model,
                            optimizer,
                            loss,
                            metrics=[MacroF1Score(num_classes=4, reduce_f = get_ytrue_and_ypred),
                                     EntityF1(corpora),
                                     EntityReconstructedF1(corpora),
                                     Accuracy(num_classes=4, reduce_f = get_ytrue_and_ypred)])

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
    
    callbacks = [LossSmoothCallback(output=True), # if output is True the smooth should be positioned before all the streaming outputs
                 TimerCallback(), # This callback should be positioned before all the streaming outputs
                 ValidationDataCallback(test_ds_NLMCHEM, name="NLMCHEM_Test", custom_inference_f=custom_inference_f),
                 #ValidationDataCallback(test_ds_CDR_Corpus, name="CDR_Test", custom_inference_f=custom_inference_f),
                 #ValidationDataCallback(test_ds_BC5CHEM_Corpus, name="BC5CHEM_Test", custom_inference_f=custom_inference_f),
                 #ValidationDataCallback(test_ds_CHEMDNER, name="CHEMDNER_Test", custom_inference_f=custom_inference_f),
                 SaveModelCallback("best", validation_name="NLMCHEM_Test", metric_name="EntityF1", cache_folder=os.path.join(PATH_CACHE, "saved_models")),
                 SaveModelCallback("end", cache_folder=os.path.join(PATH_CACHE, "saved_models")),
                 WandBLogCallback("[Extension] Biocreative Track2 NER - finetunetrain", args, entity='bitua', additional_info = additional_info),
                 ConsoleLogCallback(), # Prints the training on the console
                 EarlyStop(),
                ]
    
    
    def train_transformation(data):
        x = {"embeddings":data["embeddings"], "attention_mask":data["attention_mask"]}
        y = tf.one_hot(data["tags_int"][:,cfg["model"]["low"]:cfg["model"]["high"]], cfg["model"]["output_classes"])
        return x, y
    
    trainer.train(training_ds, 
                  epoch,
                  steps = steps,
                  custom_data_transform_f = train_transformation,
                  callbacks=callbacks)
    

    
    