package regionalization.nga;

import heuristics.Evaluator;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class WSSEvaluator implements Evaluator<TreeIndividual> {
	private Dist<double[]> dist;
	
	public WSSEvaluator(Dist<double[]> dist ) {
		this.dist = dist;
	}
	@Override
	public double evaluate(TreeIndividual i) {		
		return DataUtils.getWithinSumOfSquares(i.toCluster(), dist);
	}
}
