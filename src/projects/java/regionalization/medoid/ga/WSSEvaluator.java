package regionalization.medoid.ga;

import java.util.List;
import java.util.Map;
import java.util.Set;

import heuristics.Evaluator;
import regionalization.medoid.MedoidRegioClustering.GrowMode;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class WSSEvaluator implements Evaluator<MedoidIndividual> {
	private Dist<double[]> dist;
	Map<double[],Set<double[]>> cm;
	List<double[]> samples;
	GrowMode dm;
	
	public WSSEvaluator(List<double[]> samples, Map<double[],Set<double[]>> cm, Dist<double[]> dist, GrowMode dm ) {
		this.dist = dist;
		this.samples = samples;
		this.cm = cm;
		this.dm = dm;
	}
	@Override
	public double evaluate(MedoidIndividual i) {		
		return DataUtils.getWithinSumOfSquares(i.toCluster(samples, cm, dist, dm), dist);
	}
}
