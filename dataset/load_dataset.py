from dataset import *


dataset_dict = {
    ### Perception Dataset ###
    "dysca-clean": DyscaDataset(dataset_name="dysca-clean", type="clean"),
 
    ### Hallucination Dataset ###
    "pope": POPEDataset(dataset_name="pope", type="adversarial"),
    "openchair": OpenCHAIRDataset(dataset_name="openchair"),
    "hqh": HQHDataset(dataset_name="hqh"),

    ### Robustness Dataset ###
    "dysca-attack": DyscaDataset(dataset_name="dysca-attack", type="attack"),
    "dysca-corruption": DyscaDataset(dataset_name="dysca-corruption", type="corruption"),
    "dysca-typographic": DyscaDataset(dataset_name="dysca-typographic", type="print_attack"),

    ### Bias Dataset ###
    "vlbold-g": VLBOLDDataset(dataset_name="vlbold-g", type="gender", sampled=True),
    "vlbold-p-emo": VLBOLDDataset(dataset_name="vlbold-p-emo", type="profession-emo", sampled=True),
    "vlbold-p-nohuman": VLBOLDDataset(dataset_name="vlbold-p-nohuman", type="profession-nohuman", sampled=True),
    "vlbold-r": VLBOLDDataset(dataset_name="vlbold-r", type="race", sampled=True),
    "vlbold-ri": VLBOLDDataset(dataset_name="vlbold-ri", type="religious_ideology", sampled=True),
    "vlbbq": VLBBQDataset(dataset_name="vlbbq"),

    ### Morality Dataset ###
    "m3oralbench-classification": M3oralBenchDataset(dataset_name="M3oralBench-classification", type='classification'),
    "m3oralbench-judge": M3oralBenchDataset(dataset_name="M3oralBench-judge", type='judge'),
    "m3oralbench-choose_image": M3oralBenchDataset(dataset_name="M3oralBench-choose_image", type='choose_image'),

    ### Toxicity Dataset ###
    "toxicity-b": ToxicityDataset(dataset_name="toxicity-b", type="bloody", sampled=True),
    "toxicity-po": ToxicityDataset(dataset_name="toxicity-po", type="porn", sampled=True),
    "toxicity-pr": ToxicityDataset(dataset_name="toxicity-pr", type="protest", sampled=True),

    ### Jailbreak Dataset ###
    "figstep": FigStepDataset(dataset_name="figstep"),
    "mmsafetybench": MMSafetyBenchDataset(dataset_name="mmsafetybench"),

    ### Privacy Dataset ###
    "vpr": VPRDataset(dataset_name="vpr"),
    "vispr_leakage": VISPRLeakageDataset(dataset_name="vispr_leakage"),
    "p2-insensitive": P2Dataset(dataset_name="p2-insensitive", type="Q_Insensitive"),
    "p2-sensitive": P2Dataset(dataset_name="p2-sensitive", type="Q_Sensitive"),
}


def load_dataset(dataset_name):
    if dataset_name in dataset_dict.keys():
        dataset = dataset_dict[dataset_name]
    else:
        raise NotImplementedError(f"{dataset_name} not implemented")
    return dataset