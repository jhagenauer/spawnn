package spawnn.gui;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uci.ics.jung.graph.Graph;

public class TrainingFinishedNG extends TrainingEvent {
	private static final long serialVersionUID = 210568799038570013L;
	
	private Map<double[],Set<double[]>> bmus;
	private Graph<double[], double[]> g;

	public TrainingFinishedNG(Object source, List<double[]> samples, Map<double[], Set<double[]>> bmus, Graph<double[], double[]> g, boolean wmc ) {
		super(source,samples,wmc);
		this.bmus = bmus;
		this.g = g;
	}

	public Map<double[], Set<double[]>> getBmus() {
		return bmus;
	}

	public Graph<double[],double[]> getGraph() {
		return g;
	}
}
