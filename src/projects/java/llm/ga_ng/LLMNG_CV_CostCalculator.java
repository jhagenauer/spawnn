package llm.ga_ng;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import ga.CostCalculator;
import llm.LLMNG;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class LLMNG_CV_CostCalculator implements CostCalculator<LLMNG_Individual> {
	
	List<double[]> samples;
	int[] fa;
	int ta;
	Dist<double[]> dist;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	
	public LLMNG_CV_CostCalculator(List<double[]> samples,int[] fa, int ta) {
		this.samples = samples;
		this.cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		this.fa = fa;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLMNG_Individual i) {
		
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
			
			/*double v = SupervisedUtils.getR2(response, samplesVal, ta);
			System.out.println(v);
			ss.addValue(1.0-v);*/
			
			ss.addValue(SupervisedUtils.getRMSE(response, samplesVal, ta));
		}
		
		double mean = ss.getMean();
		if( Double.isNaN(mean) )
			return Double.POSITIVE_INFINITY;
		return mean;
	}
}
