# Forest-cover-type-using-deep-learning
Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).  This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.  Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value.  As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4).  The Rawah and Comanche Peak areas would tend to be more typical of the overall dataset than either the Neota or Cache la Poudre, due to their assortment of tree species and range of predictive variable values (elevation, etc.) Cache la Poudre would probably be more unique than the others, due to its relatively low elevation range and species composition.

# about the dataset and features:

There are 14 continuous and 40 (cover types) binary categorical features. 

# Approach used:

I have used ANN for building the neural network.

# Train test split used:

I have used 80% data for training and 20% for testing the data

Dimensions of train data (464809, 54) and of test data is (116203, 54).

# Preprocessing of data:

I have used One hot encoder on Y_test and Y_train and Standard scaler on the X_train and X_test data.

# Model and Neural layers:

1) There are in total 3 hidden layers:

HL1 has 100 neurons with a dropout of 50% so as to randomize the learning and avoid over fitting.
HL2 has 50 neurons and
HL3 has 100 neurons, both with a 50% dropout rate as well.

All three hidden layers are activated by 'relu' or retifier function.

2) There is 1 input and 1 output layer where the input size is (54,) and output size is (7,) since there are 7 classes.

where output layer is activated by 'softmax' function.
