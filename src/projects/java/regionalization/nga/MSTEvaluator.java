package regionalization.nga;

import java.util.Map.Entry;

import heuristics.Evaluator;

import java.util.Set;

import spawnn.dist.Dist;

public class MSTEvaluator implements Evaluator<TreeIndividual> {
	private Dist<double[]> dist;
	
	public MSTEvaluator(Dist<double[]> dist ) {
		this.dist = dist;
	}

	@Override
	public double evaluate(TreeIndividual i) {
		double sum = 0;
		for( Entry<double[],Set<double[]>> e : i.getTree().entrySet() ) {
			for( double[] d : e.getValue() )
				sum += dist.dist(e.getKey(), d);
		}
		return sum;
	}
}
