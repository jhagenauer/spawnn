package llm.ga.som;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import heuristics.CostCalculator;
import llm.LLMSOM;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class LLMSOM_CV_CostCalculator implements CostCalculator<LLMSOM_Individual> {

	List<double[]> samples;
	int[] fa, ga;
	int ta;
	Dist<double[]> dist;
	List<Entry<List<Integer>, List<Integer>>> cvList;

	public LLMSOM_CV_CostCalculator(List<double[]> samples, int[] fa, int[] ga, int ta) {
		this.samples = samples;
		this.cvList = SupervisedUtils.getCVList(10, 1, samples.size());
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		this.dist = new EuclideanDist(fa);
	}

	@Override
	public double getCost(LLMSOM_Individual i) {

		SummaryStatistics ss = new SummaryStatistics();
		//for (int t_max : new int[] { 10000, 50000, 100000 }) {
		//	i.iParam.put("t_max", t_max);
			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
				List<double[]> samplesTrain = new ArrayList<double[]>();
				for (int k : cvEntry.getKey())
					samplesTrain.add(samples.get(k));

				List<double[]> samplesVal = new ArrayList<double[]>();
				for (int k : cvEntry.getValue())
					samplesVal.add(samples.get(k));

				LLMSOM llmsom = i.train(samplesTrain, fa, ga, ta, 0);

				List<Double> response = new ArrayList<>();
				for (double[] x : samplesVal)
					response.add(llmsom.present(x)[0]);

				double rmse = SupervisedUtils.getRMSE(response, samplesVal, ta);
				if (Double.isNaN(rmse))
					return Double.POSITIVE_INFINITY;
				ss.addValue(rmse);
			}
		//}
		double mean = ss.getMean();
		if (Double.isNaN(mean))
			return Double.POSITIVE_INFINITY;
		return mean;
	}
}
