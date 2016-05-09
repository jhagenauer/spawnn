package regionalization.nga;

import heuristics.Evaluator;
import regionalization.nga.tabu.CutsTabuIndividual;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class WSSCutsTabuEvaluator implements Evaluator<CutsTabuIndividual> {
	private Dist<double[]> dist;
	
	public WSSCutsTabuEvaluator(Dist<double[]> dist ) {
		this.dist = dist;
	}
	@Override
	public double evaluate(CutsTabuIndividual i) {		
		return DataUtils.getWithinSumOfSuqares(i.toCluster(), dist);
	}
}
