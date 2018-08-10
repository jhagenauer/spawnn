package spawnn.gui;

import java.util.List;
import java.util.Map;
import java.util.Set;

import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.GridPos;

public class TrainingFinishedSOM extends TrainingEvent {
	private static final long serialVersionUID = 210568799038570913L;
	
	private Map<GridPos,Set<double[]>> bmus;
	private Grid2D<double[]> grid;

	public TrainingFinishedSOM(Object source, List<double[]> samples, Map<GridPos,Set<double[]>> bmus, Grid2D<double[]> grid, boolean wmc ) {
		super(source,samples,wmc);
		this.bmus = bmus;
		this.grid = grid;
	}

	public Map<GridPos, Set<double[]>> getBmus() {
		return bmus;
	}

	public Grid2D<double[]> getGrid() {
		return grid;
	}
}
