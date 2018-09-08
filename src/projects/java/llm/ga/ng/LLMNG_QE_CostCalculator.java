package llm.ga.ng;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import heuristics.CostCalculator;
import llm.LLMNG;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;

public class LLMNG_QE_CostCalculator implements CostCalculator<LLMNG_Individual> {
	
	List<double[]> samples;
	int[] fa, ga;
	int ta;
	Dist<double[]> dist;
	
	public LLMNG_QE_CostCalculator(List<double[]> samples,int[] fa, int[] ga, int ta) {
		this.samples = samples;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLMNG_Individual i) {	
		LLMNG llmng = i.train(samples, fa, ga, ta, 0);
		Map<double[],Set<double[]>> mapping = new HashMap<double[],Set<double[]>>();
		for( double[] n : llmng.getNeurons() )
			mapping.put(n, new HashSet<>() );
		
		for( double[] x : samples ) {
			llmng.present(x);
			List<double[]> neurons = llmng.getNeurons();
			double[] n0 = neurons.get(0);
			mapping.get(n0).add(x);
		}		
		return DataUtils.getMeanQuantizationError(mapping, dist);
	}
}
