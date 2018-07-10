package llm.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ga.CostCalculator;
import llm.LLMNG;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;

public class LLM_CostCalculator implements CostCalculator<LLM_Individual> {
	
	List<double[]> samples;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	int[] fa, ga;
	int ta;
	Dist<double[]> fDist, gDist;
	
	public LLM_CostCalculator(List<double[]> samples, int[] ga, int[] fa, int ta) {
		this.samples = samples;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		
		this.cvList = SupervisedUtils.getCVList(10, 1, samples.size());

		this.fDist = new EuclideanDist(fa);
		this.gDist = new EuclideanDist(ga);
	}

	@Override
	public double getCost(LLM_Individual i) {
		DescriptiveStatistics rmse = new DescriptiveStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			
			List<double[]> samplesTrain = new ArrayList<double[]>();
			for (int k : cvEntry.getKey()) {
				double[] d = samples.get(k);
				samplesTrain.add( Arrays.copyOf( d, d.length ) );
			}

			List<double[]> samplesTest = new ArrayList<double[]>();
			for (int k : cvEntry.getValue()) {
				double[] d = samples.get(k);
				samplesTest.add( Arrays.copyOf(d, d.length ) );
			}

			DataUtils.zScoreColumns( samplesTrain, samplesTest, fa );

			LLMNG llmng = i.buildModel( samplesTrain, ga, fa, ta );

			List<double[]> response = new ArrayList<>();
			List<double[]> desired = new ArrayList<>();
			for (double[] d : samplesTest) {
				response.add( llmng.present( d ) );
				desired.add( new double[] { d[ta] } );
			}
			rmse.addValue( SupervisedUtils.getRMSE( response, desired ) );
		}
		double mean = rmse.getMean();
		return Double.isNaN(mean) ? Double.POSITIVE_INFINITY : mean;
	}
}
