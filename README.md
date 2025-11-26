```markdown
# Waste Classifier 

THIS PTOJECT STILL UNDER DEVELOPMENT (77.2% VAL ACCURACY)

 uses TensorFlow / Keras with transfer learning and provides scripts to:

- Prepare Dataset
- Train a classifier (transfer learning with MobileNetV2)
- Run inference on single images and map predicted classes to organic/inorganic

Contents:
- src/train.py — training script
- src/test.py — testing script
- model - training model
- scripts/download_trashnet.py - download from kaggle 
- scripts/prepare_dataset.py - Split data
- requirements.txt — Python dependencies

Dataset:
- Recommended: TrashNet (Kaggle) or any dataset with directory structure:
  dataset/
    class_1/
      img1.jpg
      ...
    class_2/
      ...

Usage (basic):

1. Create virtualenv
   python -m venv venv
   source venv/bin/activate

2. Install dependencies
   pip install -r requirements.txt

3. Setup Kaggle API to local

4. Run scripts/download_transhnet.py and scripts/prepare_dataset.py

In terminal venv, run this:

1. python .\src\train2.py --data_dir data/train --epochs 20 --batch_size 32 --model_out model/model2.h5

2. python .\src\test.py --model model/model2.h5 --img data/val  or  python ./src/test.py --model model/model2.h5 --img https://healthscopemag.com/wp-content/uploads/2013/05/Plastic.FI_.jpg or python .\src\test.py --model model/model2.h5 --img data/val/trash/trash4.jpg

Note: just adjust the file path or parameter or img url if you train another model 





