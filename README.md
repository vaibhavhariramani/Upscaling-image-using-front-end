# SRCNN Training and Image Upscaling using Pytorch

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


# Now let's Configuring Django Server

## I have created a Django Server for Front-End Application of Image Upscaling to execute it we can to make some change

firstly, open project repo
then get into django configuration directory
`cd django_script`

change the path to external script in external script execution section in `/django_script/helloworld/views.py`

in #section - 1 put complete path of directory to `external_script_for_django` including your system path

in #section - 2 put complete path of directory to `external_script_for_django` including your system path

if everything goes well you are good to go
run command `python3 manage.py runserver 127.0.0.1:8000`
To start django then just upload image and voila script is applied on image


