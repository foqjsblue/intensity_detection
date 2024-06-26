**Stage 2 : DGCNN-based Classification**

This code is inspired by and builds upon the methods described in the [WangYueFt/dgcnn repository](https://github.com/WangYueFt/dgcnn).

If you want to use a pre-trained model, unzip the file in the checkpoints folder and specify the path in the model_path option.

**Training Mode**

Modify main.py to set the folder containing real objects and fake objects as the input path.
Ensure to correctly label them as 'Normal Object' and 'Fake Object' according to the number and order of the input folders.

```bash
python main.py --exp_name=dgcnn_256_sgd_eval --model=dgcnn --k=4 --num_points=256 --epochs=300
```

**Testing Mode**

In Testing mode, modify main.py to set the folder containing real objects and fake objects as the input path, as in Training mode.
Ensure to correctly label them as 'Normal Object' and 'Fake Object' according to the number and order of the input folders.

```bash
python main.py --exp_name=dgcnn_256_sgd_eval --model=dgcnn --k=4 --num_points=256 --epochs=300 --eval=Ture --model_path=checkpoints/dgcnn_0510_sgd/models/model.t7
```
