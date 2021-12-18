## DeepM

> A deep learning-based approach to measuring code maintainability

This repository contains materials on our paper "Measuring Code Maintainability with Deep Neural Networks".

#### In `/Datasets/`, we provide:

- two datasets:
  - TrainingData: an automatically constructed large dataset, which contains 1,394,514 Java classes.
  - TestingData: an manually constructed small dataset, which contains 240 Java classes.

#### In `/Parser/`, we provide:

- source code (written in Java) that is used to extract featurs from Java classes.
  - Based on the code, you can extract features for a single Java class or a directory that contains a number of Java classes quickly because of multithreading.

#### In `/Model/`, we provide:

- source code (`/Model/Model/*.py`):

  - `/Model/Model/GenerateDataset.py` generates TFRecords file in TensorFlow based on the training data.
  - `/Model/Model/TrainOrTest.py` trains and tests our deep learning-based model.
  - `/Model/Model/DeepM.py` computes the maintainability index of a Java class using the trained model.
  - `/Model/Model/Model.py` contains the model of DeepM, and models for ablation study.
- baselines

  - `/Model/CHC/`contains the code for the baseline CHC, which is slightly modified from [KEEL](https://github.com/SCI2SUGR/KEEL) for evaluation
  - `/Model/LRR/`contains the code for the baseline LRR, which is slightly modified from [KEEL](https://github.com/SCI2SUGR/KEEL) for evaluation
- a trained model (in `/Model/Model/all/`)
- `/Model/requirements.txt`, which describes the requirements of running our code in `/Model/Model/`.
- `/Model/runDeepM.sh`, which computes the maintainability indexes of Java classes in a directory using the traned model.

  - usage:

  ```bash
  sh runDeepM.sh [The directory containing /Model/Model/DeepM.py] [The directory containing Java classes] [The file saving results]
  ```

#### In `/Results`, we provide:

- the maintainability indexes (computed by the trained model) of Java classes in the testing data

### Contact

If you have questions, please contact me directly: `ymhu@bit.edu.cn`. Thank you!
