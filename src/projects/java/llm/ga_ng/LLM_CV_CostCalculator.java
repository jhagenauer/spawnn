package llm.ga_ng;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import ga.CostCalculator;
import llm.LLMNG;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class LLM_CV_CostCalculator implements CostCalculator<LLM_Individual> {
	
	List<double[]> samples;
	int[] fa;
	int ta;
	Dist<double[]> dist;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	
	public LLM_CV_CostCalculator(List<double[]> samples, Map<Integer,Set<double[]>> cl, int[] fa, int ta) {
		this.samples = samples;
		this.cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		this.fa = fa;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLM_Individual i) {
		
		SummaryStatistics ss = new SummaryStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			
			List<double[]> samplesTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) 
				samplesTrain.add(samples.get(k));
						
			List<double[]> samplesVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) 
				samplesVal.add(samples.get(k));
			
			LLMNG llmng = i.train(samplesTrain, fa, ta, 0);
			
			List<Double> response = new ArrayList<>();
			for( double[] x : samplesVal )
				response.add( llmng.present(x)[0] );
			
			double rmse = SupervisedUtils.getRMSE(response, samplesVal, ta);	
			ss.addValue(rmse);
		}		
		return ss.getMean();
	}
}
