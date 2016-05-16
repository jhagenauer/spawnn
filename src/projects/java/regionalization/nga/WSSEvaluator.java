package regionalization.nga;

import heuristics.Evaluator;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class WSSEvaluator implements Evaluator<TreeIndividual> {
	private Dist<double[]> dist;
	private boolean redcap = false;
	
	public WSSEvaluator(Dist<double[]> dist ) {
		this.dist = dist;
	}
	
	public WSSEvaluator(Dist<double[]> dist, boolean redcap ) {
		this.dist = dist;
		this.redcap = redcap;
	}
	
	@Override
	public double evaluate(TreeIndividual i) {	
		if( redcap )
			return DataUtils.getWithinSumOfSquares(i.toClusterRedcap(dist), dist);
		else
			return DataUtils.getWithinSumOfSquares(i.toClusterCuts(), dist);
	}
}
