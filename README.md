# SRCNN using Pytorch

## To train model :
`python train.py --zoom_factor 2 --nb_epoch 200 --cuda`

or 

run `SRCNN_model_training_using_pytorch.ipynb` file

At each epoch, a .pth model file will be saved at models folder.


## To use the model on an image: 

As we got best result with model trained at last epoch that is
`Epoch 199. Training loss: 0.0024051234012447474
Average PSNR: 26.383436188634025 dB.
`

so to execute script on single image just run command
`python run.py --image example.jpg `



