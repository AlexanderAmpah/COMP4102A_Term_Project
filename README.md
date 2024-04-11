"# COMP4102A_Term_Project" 

To run the code the code first enter "python main.py" 
then filepath of the image "images/test_boxing_27.jpg" followed by "-l" to load the model weights 
then enter the model weights file "model_weights2.weights.h5"

All together the command should look like this "python main.py images/test_boxing_27.jpg -l model_weights2.weights.h5" 

If you want to train and test the neural network yhe command should look like this 
Train: "python main.py images/test_boxing_27.jpg --train -s model_weights2.weights.h5"
Test: "python main.py images/test_boxing_27.jpg --test -l model_weights2.weights.h5"

For the neural network to do its best work handwritten text must be well lit and printed. 
Adequate spacing must be between the letters such that the letters are neither connected nor is one letter 'over' another.
