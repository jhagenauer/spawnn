# Example usage of SPAWNN toolkit

#### 1. Download shapefile with example data

Download [this](https://drive.google.com/open?id=1iTi1yzpm9yIOMDhm6gPUVeDGSp1lLaNL) shapefile and extract the contents.

#### 2. Download the toolkit

Download a pre-build jar-file of the toolkit [here](https://github.com/jhagenauer/spawnn/releases/download/0.1.10/spawnn-0.1.10.jar).            

#### 3. Start the toolkit

Start the toolkit by double-clicking on the downloaded file.

#### 4. Load the data

Click on 'Load data' and select the shapefile.

#### 5. Prepare spatial coordinates

The shapefile consists only of areal objects/counties with socioeconomic data but spatial models of SPAWNN require also coordinate features.
To automatically generate these features, click on 'Add centroid coordinates'.

#### 6. Select additional features.

Mark in the 'Use' column some interesting features like 'BLACK', 'MEDAGE2000', and PEROVER65.

#### 7. Configure the artificial neural network.

Open the 'ANN' tab. Here you can chose the type of self-organizing network (SOM or NG), the spatial context model and the corresponding parameters. For a first run, it is recommended to select 'SOM' and 'Weighted' and to leave the rest of the parameters as they are.

#### 8. Train the artifical neural network.

To finally build the network, click on 'train'. A result tab shows up which gives you a representation of the network and a map. You can interactively analyze the network, inspect selected neurons/regions, visualize the relationships between features/neurons as well as export the network and data in different formats.


      
