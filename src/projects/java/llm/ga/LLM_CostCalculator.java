package llm.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ga.CostCalculator;
import llm.LLMNG;
import llm.LLM_Lucas_CV.function;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class LLM_CostCalculator implements CostCalculator<LLM_Individual> {
	
	SpatialDataFrame sdf;
	List<Entry<List<Integer>, List<Integer>>> cvList;
	int[] fa, ga;
	int ta;
	Dist<double[]> fDist, gDist;
	
	public LLM_CostCalculator() {
		sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/lucas/lucas.shp"), true);
		cvList = SupervisedUtils.getCVList(10, 1, sdf.size());

		fa = new int[] { 
				2, // TLA
				3, // beds
				9, // rooms
				10, // lotsize
				19, // age
		};
		ga = new int[] { 20, 21 };
		ta = 0;

		for (double[] d : sdf.samples) {
			d[19] = Math.pow(d[19], 2);
			d[10] = Math.log(d[10]);
			d[2] = Math.log(d[2]);
			d[1] = Math.log(d[1]);
		}

		fDist = new EuclideanDist(fa);
		gDist = new EuclideanDist(ga);
	}

	@Override
	public double getCost(LLM_Individual i) {
		
		Sorter<double[]> sorter = new KangasSorter<>(gDist, fDist, (int)i.iParam.get("l") );
		
		DescriptiveStatistics rmse = new DescriptiveStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			
			Random r1 = new Random(0);

			List<double[]> samplesTrain = new ArrayList<double[]>();
			for (int k : cvEntry.getKey()) {
				double[] d = sdf.samples.get(k);
				samplesTrain.add( Arrays.copyOf( d, d.length ) );
			}

			List<double[]> samplesTest = new ArrayList<double[]>();
			for (int k : cvEntry.getValue()) {
				double[] d = sdf.samples.get(k);
				samplesTest.add( Arrays.copyOf(d, d.length ) );
			}

			DataUtils.zScoreColumns( samplesTrain, samplesTest, fa );

			List<double[]> neurons = new ArrayList<>();
			while (neurons.size() < LLM_Individual.nrNeurons ) {
				int idx = r1.nextInt(sdf.samples.size());
				double[] d = sdf.samples.get(idx);
				neurons.add( Arrays.copyOf(d,d.length) );
			}

			LLMNG llmng = new LLMNG(neurons, 
					getFunction( (double)i.iParam.get("nb1Init"), (double)i.iParam.get("nb1Final"), (function)i.iParam.get("nb1Func")),
					getFunction( (double)i.iParam.get("lr1Init"), (double)i.iParam.get("lr1Final"), (function)i.iParam.get("lr1Func")),
					getFunction( (double)i.iParam.get("nb2Init"), (double)i.iParam.get("nb2Final"), (function)i.iParam.get("nb2Func")),
					getFunction( (double)i.iParam.get("lr2Init"), (double)i.iParam.get("lr2Final"), (function)i.iParam.get("lr2Func")),
					sorter, fa, 1);
			llmng.aMode = (LLMNG.mode)i.iParam.get("mode");

			for (int t = 0; t < LLM_Individual.t_max; t++) {
				int j = r1.nextInt( samplesTrain.size() );
				double[] d = samplesTrain.get(j);
				llmng.train((double) t / LLM_Individual.t_max, d, new double[] { d[ta] });
			}

			List<double[]> response = new ArrayList<>();
			List<double[]> desired = new ArrayList<>();
			for (double[] d : samplesTest) {
				response.add( llmng.present( d ) );
				desired.add( new double[] { d[ta] } );
			}
			rmse.addValue( SupervisedUtils.getRMSE( response, desired ) );
		}
		double mean = rmse.getMean();
		return Double.isNaN(mean) ? Double.MAX_VALUE : mean;
	}
	
	private DecayFunction getFunction(double init, double fin, function func) {
		if (func == function.Power)
			return new PowerDecay(init, fin);
		else if (func == function.Linear)
			return new LinearDecay(init, fin);
		else
			throw new RuntimeException("Unkown function");
	}
}
