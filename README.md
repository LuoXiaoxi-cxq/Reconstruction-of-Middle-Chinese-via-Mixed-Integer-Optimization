## Phonetic Reconstruction of the Consonant System of Middle Chinese via Mixed Integer Optimization

This paper has been accepted to TACL. The preprint version is [here](https://www.arxiv.org/abs/2502.04625).

### requirments
```
gurobipy==12.0.0
numpy==1.22.4
pandas==1.4.2
scikit_learn==1.3.2
```
We use `Gurobi`, a commercial optimizer, to solve our large-scale optimization problem.
To successfully run the code, in addition to the above-listed packages, you will need to obtain a [Gurobi](https://www.gurobi.com/) license and activate it.

## Code
### Experiments with synthetic data
To generate synthetic data, run:
```shell
python -m synthetic.synthetic_dataset -p_fq 0.3 -p_dia 0.3 p_char 0.3 -phon Latin  # generate data
```
This command will set `file_name` to `fq_0.3_dia_0.3_char_0.3_phon_Latin`, and generate corresponding 
`fq_{file_name}.xlsx` and `dia_{file_name}.xlsx` files for character-speller pairs and variations.
Generated data will be saved under `synthetic/result/{file_name}/`.

You can adjust various parameters (for their meaning, 
please refer to Section 5 in our paper). `p_fq`, `p_dia`, `p_dia` represent `change rate', 
deciding the degree to which generated variations deviate from their ancestors. 

Then, solve the ancestor form based on synthetic data.
```
python -m synthetic.main_synthetic -pth {file_name}  # solve optimization problem
```

### Experiments based on real data
For experiments with real data, the code is in `src/main.py`. The reconstruction of Middle Chinese will be performed,
and evaluation metrics (Adjusted Mutual Information, sound rate, and consistency on held-out data) will be automatically calculated.

| Parameter        | Shortened Name | Description                                                                                                                                                     |
|------------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fq_medial_weight | fqmw           | Weight assigned to Fanqie (反切) spellings (X, X_u) that satisfy X and X_u sharing the same medial. See Section 5.4 of out paper for details.                     |
| fanqie_weight    | w              | Weight of Fanqie information in the objective function. In our paper, $\lambda_{\text{fq}}$ equals to `w/(w+1)`.                                                |
| fanqie           | fq             | Whether to include Fanqie information in modeling.                                                                                                              |
| yidu             | yd             | Whether to include Yidu information in modeling.                                                                                                                |
| restrict         | r              | 	Indicates whether constraints designed to obtain a proper phonetic feature vector are incorporated into the model. (See Section 4.3 of our paper for details.) |
| verify_p         | vp             | Portion of held-out data used for evaluation.                                                                                                                   |
| sol              | sol            | Path to pre-solved solutions. If you have previously run experiments and saved the solutions, use this parameter to specify the path for loading them.          |                                                                                                                |


### Baseline
We report two versions of majority vote as baseline: IPA-level and feature-level. 
In IPA-level majority vote, for each character, we select the most frequent IPA phoneme from all 20 dialects as its reconstructed initial.
For feature-level majority vote, we choose the most frequent value for each feature of each character.

For synthetic data, baseline code is in `synthetic/baseline.py`. For real data (20 modern Chinese dialects),
baseline code is in `eval/baseline.py`.


## Dataset
All datasets are stored in the `data/` directory, with most data in traditional Chinese. Please refer to Section 6.1 of our paper for details.
+ `data/baseline/gy_vote_dict_{feature/IPA}.npy`: Pre-saved files for majority vote. See `eval/baseline.py`.
+ `data/IPA/MyFeatures.xlsx`: Our phoneme representation based on distinctive feature vectors. See Section 3 of our paper.
+ `data/synthetic/languages.xlsx`: Consonant systems of several languages, used by `synthetic/synthetic_dataset.py` to generate synthetic data. 
You could freely add consonant systems of new languages here.
+ `data/char_1990_2661.pickle`: Our final representative character set, with 1990 different characters and 2661 entries.
+ `data/out_charset_gt.xlsx`: Initial categories in Guangyun (廣韻) of our final character set, with 1990 characters and 2661 entries.
+ `data/align_{train/result}_initial.xlsx`: Aligned data between Guangyun and Hanyu Fangyin Zihui (漢語方音字匯), 
containing the pronunciations of our final character set in 20 modern Chinese dialects.
+ `data/fanqie_common.npy`: If a Fanqie speller has multiple pronunciations in Guangyun, the most common pronunciation used for Fanqie is recorded here.

The Hanyu Fangyin Zihui (漢語方音字匯) and Guangyun (廣韻) datasets are provided by the Department of Chinese Language and Literature, Peking University. 
Another version of Guangyun dataset can be found at [kaom.net](http://www.kaom.net/).
