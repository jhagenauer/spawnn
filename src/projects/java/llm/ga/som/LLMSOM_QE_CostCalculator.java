package llm.ga.som;

import java.util.List;

import heuristics.CostCalculator;
import llm.LLMSOM;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.utils.SomUtils;

public class LLMSOM_QE_CostCalculator implements CostCalculator<LLMSOM_Individual> {
	
	List<double[]> samples;
	int[] fa, ga;
	int ta;
	Dist<double[]> dist;
	
	public LLMSOM_QE_CostCalculator(List<double[]> samples,int[] fa, int[] ga, int ta) {
		this.samples = samples;
		this.fa = fa;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLMSOM_Individual i) {	
		LLMSOM llmng = i.train(samples, fa, ga, ta, 0);
		return SomUtils.getQuantizationError(llmng.getGrid(), llmng.getBmuGetter(), dist, samples);
	}
}
