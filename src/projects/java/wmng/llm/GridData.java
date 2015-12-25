package wmng.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import spawnn.som.grid.Grid2D;
import spawnn.utils.GeoUtils;


public class GridData {
	Map<double[], Map<double[], Double>> dMapTrain, dMapVal;
	List<double[]> samplesTrain = new ArrayList<double[]>();
	List<double[]> desiredTrain = new ArrayList<double[]>();
	List<double[]> samplesVal = new ArrayList<double[]>();
	List<double[]> desiredVal = new ArrayList<double[]>();
	Grid2D<double[]> gridTrain,gridVal;

	public GridData(Grid2D<double[]> gridTrain, Grid2D<double[]> gridVal) {
		this.gridTrain = gridTrain;
		dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridTrain)));
		for (double[] d : gridTrain.getPrototypes()) {
			samplesTrain.add(d);
			desiredTrain.add(new double[] { d[3] });
		}

		this.gridVal = gridVal;
		dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridVal)));
		for (double[] d : gridVal.getPrototypes()) {
			samplesVal.add(d);
			desiredVal.add(new double[] { d[d.length - 1] });
		}
	}
}