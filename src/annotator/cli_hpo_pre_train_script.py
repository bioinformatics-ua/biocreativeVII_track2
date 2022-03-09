import argparse
import json

import tensorflow as tf
import tensorflow_addons as tfa

import glob
import os

# import trainer
# import metrics MacroF1Score, Accuracy, MacroF1ScoreBI, EntityF1
from polus.callbacks import ConsoleLogCallback, TimerCallback, LossSmoothCallback, ValidationDataCallback, SaveModelCallback, EarlyStop, WandBLogCallback, HPOPruneCallback
from polus.utils import set_random_seed, complex_json_serializer
from polus.data import CachedDataLoader, build_bert_embeddings, DataLoader
from polus.schedulers import warmup_scheduler
from polus.ner.metrics import MacroF1Score, Accuracy
from polus.training import ClassifierTrainer
from polus.hpo import parameter, HPO_Objective

import modelsv2

from utils import get_temp_file
from data import short_checkpoint_names, bertseq_left_generator, bertseq_center_generator, tokseq_generator, sentence_generator, passage_generator, document_generator, selector_generator, bertseq_left128_generator, tokseqconcat_generator, random_augmentation, ShufflerAugmenter, NoiseAugmenter
from losses import sum_cross_entropy, weighted_cross_entropy, sample_weighted_cross_entropy
from corpora import NLMChemCorpus, CDRCorpus, CHEMDNERCorpus, DrugProtFilteredCorpus, BC5CDRCorpus, CRAFTCorpus, BioNLP11IDCorpus, BioNLP13CGCorpus, BioNLP13PCCorpus
from preprocessing import Tokenizer, PUBMEDBERT_FULL, SAPBERT
from metrics import EntityF1

from transformers.optimization_tf import WarmUp, AdamWeightDecay


def build_trainer(args):
        
    PATH_CACHE = "/backup/cache_biocreative_extension_track2/"
    
    BERT_CHECKPOINT = PUBMEDBERT_FULL
    
    def init_train():
        
        
        cfg = {
            "embeddings":{
                "type":"bert",
                "checkpoint": BERT_CHECKPOINT,
                "bert_layer_index": -3, 
            },
            "model":{
                "sequence_length":256,
                "output_classes": 4,
                "low": 128, 
                "high": 384,
                "activation": parameter("mish", lambda trial:trial.suggest_categorical('dense_activation', ["relu", "selu", "mish"])),
                "dropout_p": parameter(0.4, lambda trial:trial.suggest_float('dropout', 0.1, 0.6)),
                "hidden_space": parameter(128, lambda trial:trial.suggest_int('dense_units', 400, 1024)), 
                "gaussian_noise": parameter(args.gaussian_noise, lambda trial:trial.suggest_float('gaussian_noise', 0.01, 0.2)),
            }
        }
        
        set_random_seed()

        args.use_crf_mask = True#parameter(args.use_crf_mask, lambda trial:trial.suggest_categorical("use_crf_mask", [True, False]))
        args.random_augmentation = parameter(args.random_augmentation, lambda trial: trial.suggest_categorical("random_augmentation", ["shuffle", "noise"]))
        args.use_sample_weights = parameter(args.use_sample_weights, lambda trial: trial.suggest_categorical("use_sample_weights", [True, False]))
        args.epoch = 30#parameter(args.epoch, lambda trial: trial.suggest_int('epoch', 15, 30))
        args.base_lr = parameter(args.base_lr, lambda trial: trial.suggest_loguniform("lr", 1e-6, 1e-4))

        if args.use_crf_mask:
            cfg["model"]["mask_impossible_transitions"] = tf.constant([[1,1,1,1],[1,1,1,0],[1,1,1,1],[1,1,1,1]], dtype=tf.float32)
        else:
            cfg["model"]["mask_impossible_transitions"] = None

        corpora = [NLMChemCorpus()]

        dataloaders = {"train":[], "dev":[], "test":[]}

        check_point_name = short_checkpoint_names[cfg["embeddings"]["checkpoint"]]
        _index = cfg["embeddings"]["bert_layer_index"]
        ## Building the DataLoaders by using the 
        for group in dataloaders.keys():
            for corpus in corpora:
                if group in corpus:
                    dataloaders[group].append(CachedDataLoader.from_cached_index(os.path.join(PATH_CACHE, "dataloaders", f"index{_index}_{check_point_name}_heavy_batch_map_function_{corpus}_{group}_document_tokseqconcat_bertseq_center.index")))

        print("Clean up any heavy model that is on the GPU")
        tf.keras.backend.clear_session()

        train_dls = dataloaders["train"]+dataloaders["dev"]#+dataloaders["test"][1:]
        test_dls = [dataloaders["test"][0]]

        print("Merge the training dataloaders")
        train_dataloader = CachedDataLoader.merge(*train_dls)

        if args.random_augmentation is not None:

            n_samples = parameter(7000, lambda trial: trial.suggest_int('n_samples', 5000, 15000))#train_dataloader.get_n_samples()

            _prob_change_entity = parameter(0.3, lambda trial:trial.suggest_float('prob_change_entity', 0.1, 0.95))

            cfg["augmentation"] = {
                    "k_selections": parameter(5, lambda trial: trial.suggest_int('k_selections', 1, 10)),
                    "prob_change_entity": _prob_change_entity,
                    "prob_change_non_entity": parameter(0.3, lambda trial:trial.suggest_float('prob_change_non_entity', 0., 1-_prob_change_entity)),
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
                                          .shuffle(30000)\
                                          .batch(args.batch_size, drop_remainder=True)\
                                          .prefetch(tf.data.AUTOTUNE)

        test_ds_NLMCHEM = test_dls[0].batch(args.batch_size)\
                                     .prefetch(tf.data.experimental.AUTOTUNE)

        model = modelsv2.BERT_MLP_DROPOUT_CRF_V2(**cfg)
        _sample = next(iter(test_ds_NLMCHEM))
        model.init_from_data(**{"embeddings": _sample["embeddings"], "attention_mask": _sample["attention_mask"]})
        model.summary()

        if args.use_sample_weights:
            loss = model.loss_sample_weights([0, 0, 1, 1], 0.4)
        else:
            loss = model.loss

        epoch = args.epoch


        steps =  n_samples//args.batch_size

        optimizer = AdamWeightDecay(
            learning_rate = warmup_scheduler((steps+1)*epoch, args.base_lr),
            weight_decay_rate = 1e-3,
            exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"],
        )


        def get_ytrue_and_ypred(sample):
            return sample["tags_int"], sample["tags_int_pred"]

        trainer = ClassifierTrainer(model,
                                    optimizer,
                                    loss,
                                    metrics=[MacroF1Score(num_classes=4, reduce_f = get_ytrue_and_ypred),
                                             EntityF1(corpora),
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

        trainer.changing_train_config(tf_dataset=training_ds, 
                                      epochs=epoch,
                                      steps = steps,
                                      train_map_f = train_transformation,
                                      callbacks=callbacks)
        return trainer
    
    return init_train


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument("study_name", type=str, help="A json file with the configuration of the model")
    parser.add_argument("-n_trials", type=int, default=20, help="Number of trials to be used on the HPO")
    parser.add_argument("-base_lr", type=float, default=0.001, help="The base learning rate for the optimizer")
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size that will be used during training")
    parser.add_argument("-epoch", type=int, default=30, help="Number of epochs during train")
    parser.add_argument("-gaussian_noise", type=float, default=None, help="Gausian noise")
    parser.add_argument("-random_augmentation", type=str, default=None, help="Mode to use random data augmentation")
    parser.add_argument("-label_smoothing", type=float, default=0, help= "Amount of label smoothing to be used during cross entropy")
    parser.add_argument("-use_dice_loss", action='store_true', help= "Flag that replaces the cross entropy loss by the dice_loss")
    parser.add_argument("-use_sample_weights", action='store_true', help= "Flag that enables the training with sample weights")
    parser.add_argument("-use_class_weights", action='store_true', help= "Flag that enables the training with class weights")
    parser.add_argument("-use_crf_mask", action='store_true', help="Flag that enables the masking of impossible transitions on the CRF")
    parser.add_argument("-use_fulltext", action='store_true', help="Flag to use the fulltext of the corpus where the passages are discarded")
    
    args = parser.parse_args()
    
    init_trainer = build_trainer(args)
 
    with HPO_Objective(init_trainer, "NLMCHEM_Test", "EntityF1") as obj:
        print("HPO backend", obj.backend)
        
        # change the default cfg
        obj.change_optuna_config(n_trials=args.n_trials,
                                 study_name=args.study_name)
        
        
        print("Optuna config")
        for k,v in obj.optuna_cfg.items():
            print("\t",k,":",v)
        

        print("Pruning", obj.add_pruning_cb)

        
    
    print(obj.study.best_trial)
    
    #trainer = init_trainer()
    #trainer.train()
