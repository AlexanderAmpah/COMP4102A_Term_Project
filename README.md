"# COMP4102A_Term_Project" 

Authors:

Ainan Kashif        101197764
Oko Ampah           101169097
Emmanuel Richmond   101108167 

Date: Winter Semester 2024 

GitHub Repo:    https://github.com/AlexanderAmpah/COMP4102A_Term_Project

To run the code the code first enter "python main.py", then filepath of the image "images/test_boxing.jpg", followed by "-l" to load the model weights. Model weights files must end with a ".weights.h5" extension.

Example: "python main.py images/test_boxing_30.jpg -l model_weights2.weights.h5" 

To train the model run: "python main.py images/test_boxing.jpg --train"

If you want to train or test the neural network while saving the weights the commands should look like this:

Train: "python main.py images/test_boxing_27.jpg ---train -s model_weights2.weights.h5"
Test: "python main.py images/test_boxing_27.jpg ---test -l model_weights2.weights.h5"

For the neural network to do its best work handwritten text must be well lit and printed. 
Adequate spacing must be between the letters such that the letters are neither connected nor is one letter 'over' another.
