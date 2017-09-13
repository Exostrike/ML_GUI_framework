# ML_GUI_framework
General purpose framework for doing binary classification prediction using python's Sklearn. Able to lable a dataset and print the labeled dataset for later use. 

![alt text](https://i.imgur.com/9WcZ9Qx.png)

While you can use a number of preset options you also have the option of chosing a specific ensemlbe method including the models to be used in a hetrogeneous manner.

![alt text](https://i.imgur.com/aeYsQcl.png)

##Operating instructions:

It is controlled with a GUI which can be activated with the command line command:

python gui.py


Once loaded this GUI allows you to load the required file, select target variable, set binary classification boundary, remove unwanted observations based on target variable. Set secondary target variables and drop unwanted variables. The resulting labeled dataset is automatically printed to a CSV filed called 'dateset' in the directory location. When models are run results are printed to text file called 'results'. ROC curve matplotlib plots are saved to the directory.

##Required Packages:

sklearn<br/><br/>
scipy<br/><br/>
numpy<br/><br/>
pandas<br/><br/>
itertools<br/><br/>
mlxtend<br/><br/>
imblearn<br/><br/>
collections<br/><br/>
wx<br/><br/>
sys<br/><br/>
